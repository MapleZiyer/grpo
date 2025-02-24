import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import numpy as np

class HoverDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: T5Tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Dict[str, str]]:
        # 加载Hover数据集
        # 这里需要根据实际的数据格式进行调整
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 处理输入文本和目标文本
        input_text = item['input_text']
        target_text = item['target_text']
        
        # 编码输入
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码目标
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.input_ids.squeeze(),
            'raw_input': input_text,
            'raw_target': target_text
        }

class GRPO:
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        reward_model: nn.Module,
        tokenizer: T5Tokenizer,
        learning_rate: float = 1e-5,
        eps_clip: float = 0.2
    ):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.eps_clip = eps_clip

    def compute_rewards(self, generated_outputs: List[str], reference_outputs: List[str]) -> torch.Tensor:
        # 使用奖励模型计算奖励值
        with torch.no_grad():
            rewards = self.reward_model(generated_outputs, reference_outputs)
        return rewards

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        # 生成序列
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch['input_ids'].unsqueeze(0),
                attention_mask=batch['attention_mask'].unsqueeze(0),
                max_length=self.max_length,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # 解码生成的序列
        generated_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        reference_texts = [batch['raw_target']]
        
        # 计算奖励
        rewards = self.compute_rewards(generated_texts, reference_texts)
        
        # 计算策略梯度
        old_log_probs = outputs.scores[0].log_softmax(dim=-1)
        new_outputs = self.model(
            input_ids=batch['input_ids'].unsqueeze(0),
            attention_mask=batch['attention_mask'].unsqueeze(0),
            labels=outputs.sequences
        )
        new_log_probs = new_outputs.logits.log_softmax(dim=-1)
        
        # 计算比率
        ratio = (new_log_probs - old_log_probs).exp()
        
        # 计算裁剪后的目标
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * rewards
        loss = -torch.min(surr1, surr2).mean()
        
        # 优化器步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "reward": rewards.mean().item()
        }

def main():
    # 初始化模型和tokenizer
    model_name = "t5-base"  # 或其他T5模型变体
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 加载奖励模型（需要根据实际情况修改）
    reward_model = torch.load('path_to_reward_model.pt')
    
    # 初始化数据集
    train_dataset = HoverDataset(
        data_path='path_to_hover_dataset.json',
        tokenizer=tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # GRPO通常使用小批量
        shuffle=True
    )
    
    # 初始化训练器
    trainer = GRPO(
        model=model,
        reward_model=reward_model,
        tokenizer=tokenizer
    )
    
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        total_reward = 0
        for batch in train_dataloader:
            # 移动数据到设备
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # 训练步骤
            metrics = trainer.train_step(batch)
            total_loss += metrics['loss']
            total_reward += metrics['reward']
        
        # 打印训练信息
        avg_loss = total_loss / len(train_dataloader)
        avg_reward = total_reward / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Reward: {avg_reward:.4f}')
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            model.save_pretrained(f'model_checkpoint_epoch_{epoch+1}')

if __name__ == "__main__":
    main()