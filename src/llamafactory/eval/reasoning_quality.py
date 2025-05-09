# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re  
import math  
from typing import List, Dict, Any, Optional, Union, Tuple  
import torch  
from torch import nn
import numpy as np  
  
class ReasoningQualityEvaluator(nn.Module):  
    """  
    Evaluates reasoning quality for DPO training by incorporating GRPO reward functions.  
      
    This evaluator combines multiple reward functions from GRPO:  
    1. Format reward - checks if the text follows the <think>...</think><answer>...</answer> format  
    2. Accuracy reward - can be used if ground truth is available  
    3. Cosine reward - controls length based on correctness  
    4. Repetition penalty - penalizes repetitive content  
    5. Soft overlong punishment - penalizes excessively long responses  
    """  
      
    def __init__(  
        self,  
        tokenizer=None,  
        # Format reward parameters  
        format_weight: float = 0.3,  
          
        # Cosine reward parameters  
        cosine_weight: float = 0.3,  
        cosine_min_len_value_wrong: float = -0.5,  
        cosine_max_len_value_wrong: float = 0.0,  
        cosine_min_len_value_correct: float = 1.0,  
        cosine_max_len_value_correct: float = 0.5,  
        cosine_max_len: Optional[int] = None,  
          
        # Repetition penalty parameters  
        repetition_weight: float = 0.2,  
        repetition_n_grams: int = 3,  
        repetition_max_penalty: float = -1.0,  
          
        # Soft overlong parameters  
        overlong_weight: float = 0.2,  
        soft_max_length: Optional[int] = None,  
        soft_cache_length: Optional[int] = None  
    ):  
        super().__init__()  
        self.tokenizer = tokenizer  

        # 初始化用于存储最近一次评估的各个组件得分  
        self.last_component_scores = {}
          
        # Weights for different reward components  
        self.format_weight = format_weight  
        self.cosine_weight = cosine_weight  
        self.repetition_weight = repetition_weight  
        self.overlong_weight = overlong_weight  
          
        # Cosine reward parameters  
        self.cosine_min_len_value_wrong = cosine_min_len_value_wrong  
        self.cosine_max_len_value_wrong = cosine_max_len_value_wrong  
        self.cosine_min_len_value_correct = cosine_min_len_value_correct  
        self.cosine_max_len_value_correct = cosine_max_len_value_correct  
        self.cosine_max_len = cosine_max_len or 2048  # Default max length  
          
        # Repetition penalty parameters  
        self.repetition_n_grams = repetition_n_grams  
        self.repetition_max_penalty = repetition_max_penalty  
          
        # Soft overlong parameters  
        self.soft_max_length = soft_max_length or 2048  
        self.soft_cache_length = soft_cache_length or 512  
      
    def forward(self, chosen_texts: List[str], rejected_texts: List[str], ground_truths: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:  
        """  
        评估chosen和rejected文本的推理质量。  
        """  
        chosen_scores = []  
        rejected_scores = []  
        
        # 用于存储各个组件的得分  
        chosen_format_scores = []  
        chosen_cosine_scores = []  
        chosen_repetition_scores = []  
        chosen_overlong_scores = []  
        
        rejected_format_scores = []  
        rejected_cosine_scores = []  
        rejected_repetition_scores = []  
        rejected_overlong_scores = []  
        
        for i, (chosen, rejected) in enumerate(zip(chosen_texts, rejected_texts)):  
            ground_truth = ground_truths[i] if ground_truths else None  
            
            # 计算chosen文本的各个组件得分  
            chosen_format_score = self._evaluate_format(chosen)  
            chosen_repetition_score = self._evaluate_repetition(chosen)  
            chosen_overlong_score = self._evaluate_overlong(chosen)  
            
            # 计算rejected文本的各个组件得分  
            rejected_format_score = self._evaluate_format(rejected)  
            rejected_repetition_score = self._evaluate_repetition(rejected)  
            rejected_overlong_score = self._evaluate_overlong(rejected)  
            
            # 如果有ground truth，计算准确性和余弦奖励  
            if ground_truth:  
                chosen_is_correct = self._is_correct(chosen, ground_truth)  
                rejected_is_correct = self._is_correct(rejected, ground_truth)  
                
                chosen_cosine_score = self._evaluate_cosine(chosen, chosen_is_correct)  
                rejected_cosine_score = self._evaluate_cosine(rejected, rejected_is_correct)  
            else:  
                # 如果没有ground truth，假设chosen是"更正确的"  
                chosen_cosine_score = self._evaluate_cosine(chosen, True)  
                rejected_cosine_score = self._evaluate_cosine(rejected, False)  
            
            # 收集各个组件的得分  
            chosen_format_scores.append(chosen_format_score)  
            chosen_cosine_scores.append(chosen_cosine_score)  
            chosen_repetition_scores.append(chosen_repetition_score)  
            chosen_overlong_scores.append(chosen_overlong_score)  
            
            rejected_format_scores.append(rejected_format_score)  
            rejected_cosine_scores.append(rejected_cosine_score)  
            rejected_repetition_scores.append(rejected_repetition_score)  
            rejected_overlong_scores.append(rejected_overlong_score)  
            
            # 计算总得分  
            chosen_total = (  
                self.format_weight * chosen_format_score +  
                self.cosine_weight * chosen_cosine_score +  
                self.repetition_weight * chosen_repetition_score +  
                self.overlong_weight * chosen_overlong_score  
            )  
            
            rejected_total = (  
                self.format_weight * rejected_format_score +  
                self.cosine_weight * rejected_cosine_score +  
                self.repetition_weight * rejected_repetition_score +  
                self.overlong_weight * rejected_overlong_score  
            )  
            
            chosen_scores.append(chosen_total)  
            rejected_scores.append(rejected_total)  
        
        # 保存各个组件的得分，用于记录指标  
        self.last_component_scores = {  
            "format_chosen": torch.tensor(chosen_format_scores),  
            "format_rejected": torch.tensor(rejected_format_scores),  
            "cosine_chosen": torch.tensor(chosen_cosine_scores),  
            "cosine_rejected": torch.tensor(rejected_cosine_scores),  
            "repetition_chosen": torch.tensor(chosen_repetition_scores),  
            "repetition_rejected": torch.tensor(rejected_repetition_scores),  
            "overlong_chosen": torch.tensor(chosen_overlong_scores),  
            "overlong_rejected": torch.tensor(rejected_overlong_scores),  
        }  
        
        return torch.tensor(chosen_scores), torch.tensor(rejected_scores)
      
    def _evaluate_format(self, text: str) -> float:  
        """  
        Evaluate if the text follows the <think>...</think><answer>...</answer> format.  
        Based on the Format reward function in GRPO.  
          
        Returns:  
            1.0 if format is correct, 0.0 otherwise  
        """  
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'  
        match = re.search(pattern, text, re.DOTALL)  
          
        if not match:  
            return 0.0  
          
        # Check if there's meaningful content in both think and answer sections  
        think_content = match.group(1).strip()  
        answer_content = match.group(2).strip()  
          
        if len(think_content) < 10 or len(answer_content) < 2:  
            return 0.0  
              
        return 1.0  
      
    def _is_correct(self, text: str, ground_truth: str) -> bool:  
        """  
        Simple accuracy check to determine if the answer is correct.  
        In a real implementation, this would use more sophisticated methods.  
          
        Returns:  
            True if the answer appears to be correct, False otherwise  
        """  
        # Extract answer part  
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)  
        if not answer_match:  
            return False  
          
        answer = answer_match.group(1).strip()  
          
        # Simple string matching - in practice, use more sophisticated methods  
        # like those in the accuracy reward function in GRPO  
        return answer in ground_truth or ground_truth in answer  
      
    def _evaluate_cosine(self, text: str, is_correct: bool) -> float:  
        """  
        Cosine reward function based on GRPO implementation.  
        - For correct answers: rewards decrease as length increases  
        - For incorrect answers: rewards increase as length increases  
          
        Returns:  
            A float reward value  
        """  
        # Get text length (token count if tokenizer is available)  
        if self.tokenizer:  
            length = len(self.tokenizer.encode(text))  
        else:  
            length = len(text.split())  
          
        # Normalize length to [0, 1] range  
        normalized_length = min(length / self.cosine_max_len, 1.0)  
          
        if is_correct:  
            # For correct answers: reward decreases with length  
            min_val = self.cosine_max_len_value_correct  
            max_val = self.cosine_min_len_value_correct  
        else:  
            # For incorrect answers: reward increases with length  
            min_val = self.cosine_min_len_value_wrong  
            max_val = self.cosine_max_len_value_wrong  
          
        # Apply cosine function for smooth transition  
        cosine_value = 0.5 * (1 + math.cos(normalized_length * math.pi))  
        reward = min_val + (max_val - min_val) * cosine_value  
          
        return reward  
      
    def _evaluate_repetition(self, text: str) -> float:  
        """  
        Repetition penalty based on GRPO implementation.  
        Penalizes repetitive n-grams in the text.  
          
        Returns:  
            A float penalty value (negative or zero)  
        """  
        words = text.lower().split()  
        if len(words) < self.repetition_n_grams:  
            return 0.0  
          
        # Extract n-grams  
        ngrams = []  
        for i in range(len(words) - self.repetition_n_grams + 1):  
            ngram = tuple(words[i:i + self.repetition_n_grams])  
            ngrams.append(ngram)  
          
        # Calculate repetition ratio  
        unique_ngrams = set(ngrams)  
        if not ngrams:  
            return 0.0  
              
        repetition_ratio = 1 - len(unique_ngrams) / len(ngrams)  
          
        # Apply penalty  
        penalty = repetition_ratio * self.repetition_max_penalty  
          
        return penalty  
      
    def _evaluate_overlong(self, text: str) -> float:  
        """  
        Soft overlong punishment based on GRPO implementation.  
        Applies a linear penalty for responses exceeding a threshold length.  
          
        Returns:  
            A float penalty value (negative or zero)  
        """  
        # Get text length (token count if tokenizer is available)  
        if self.tokenizer:  
            length = len(self.tokenizer.encode(text))  
        else:  
            length = len(text.split())  
          
        # Calculate threshold  
        threshold = self.soft_max_length - self.soft_cache_length  
          
        # If length is below threshold, no penalty  
        if length <= threshold:  
            return 0.0  
          
        # If length exceeds max length, maximum penalty  
        if length >= self.soft_max_length:  
            return -1.0  
          
        # Linear penalty in the interval [threshold, soft_max_length]  
        penalty_ratio = (length - threshold) / self.soft_cache_length  
        penalty = -penalty_ratio  
          
        return penalty  
      
    def evaluate_reasoning_quality(self, text: str) -> Dict[str, float]:  
        """  
        Comprehensive evaluation of reasoning quality for a single text.  
        Useful for analysis and debugging.  
          
        Returns:  
            Dictionary with individual reward components  
        """  
        format_score = self._evaluate_format(text)  
        repetition_score = self._evaluate_repetition(text)  
        overlong_score = self._evaluate_overlong(text)  
          
        # For cosine score, we need to know if it's correct  
        # Without ground truth, we can't determine this accurately  
        cosine_score_if_correct = self._evaluate_cosine(text, True)  
        cosine_score_if_incorrect = self._evaluate_cosine(text, False)  
          
        return {  
            "format_score": format_score,  
            "repetition_score": repetition_score,  
            "overlong_score": overlong_score,  
            "cosine_score_if_correct": cosine_score_if_correct,  
            "cosine_score_if_incorrect": cosine_score_if_incorrect,  
            "total_if_correct": (  
                self.format_weight * format_score +  
                self.cosine_weight * cosine_score_if_correct +  
                self.repetition_weight * repetition_score +  
                self.overlong_weight * overlong_score  
            ),  
            "total_if_incorrect": (  
                self.format_weight * format_score +  
                self.cosine_weight * cosine_score_if_incorrect +  
                self.repetition_weight * repetition_score +  
                self.overlong_weight * overlong_score  
            )  
        }