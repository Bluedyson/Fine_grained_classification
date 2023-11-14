"""
coding:utf-8
@Time:2022/12/17 18:29
"""
from torch.utils.data.dataset import T_co
import torch
import numpy as np
import random
import time
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertTokenizer, AdamW


def set_seed(seed=123):
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed_all(seed)


class Args:
	def __init__(self, label2id: dict):
		self.label2id = {}
		self.model_path = r"G:\Search\bigData\NLP_pyabsa\prompt_SENTI\model\chinese_roberta_wwm_ext"
		self.max_seq_len = 100
		self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
		self.batch_size = 32
		self.weight_decay = 0.01
		self.epochs = 1
		self.learning_rate = 7e-6
		self.total_step = 0
		self.eval_epochs = 0.5

		self.best_acc = 0.
		self.prompt = ""

		self.ckpt_path = "output/{}.pt".format(time.ctime().replace(":", " ").replace(" ", "_"))
		self.tokenizer = BertTokenizer.from_pretrained(self.model_path)

		self.label2id = label2id
		self.tokenizer.add_special_tokens({"additional_special_tokens": [f"[unused{i}]" for i in range(1, 50)]}, True)
		self.label_ids = [self.tokenizer.convert_tokens_to_ids(key) for key in self.label2id.keys()]

	def toString(self):
		return "\n".join([
			str(key) + ": " + str(value) for key, value in self.__dict__.items()
		])


def pprint(text: str):
	print(time.strftime("%H:%M:%S") + " | " + text)


class BertDataset(Dataset):
	def __len__(self):
		return len(self.data)

	def __getitem__(self, index) -> T_co:
		return self.data[index]

	def __init__(self,
	             args,
	             load_data_fn: callable):
		self.tokenizer = args.tokenizer
		self.max_seq_len = args.max_seq_len
		self.prompt = args.prompt
		self.label_ids = args.label_ids
		self.load_data_fn = load_data_fn
		self.data = []

	def collate_fn(self, batch: [[str, str]]):
		input_ids_all = []
		token_type_ids_all = []
		attention_mask_all = []
		for text in batch:
			inputs = self.tokenizer(text=text,
			                        max_length=self.max_seq_len,
			                        padding="max_length",
			                        truncation="longest_first",
			                        return_attention_mask=True,
			                        return_token_type_ids=True)

			input_ids_all.append(inputs["input_ids"])
			token_type_ids_all.append(inputs["token_type_ids"])
			attention_mask_all.append(inputs["attention_mask"])

		return {
			"input_ids": torch.tensor(input_ids_all, dtype=torch.long),
			"attention_mask": torch.tensor(attention_mask_all, dtype=torch.long),
			"token_type_ids": torch.tensor(token_type_ids_all, dtype=torch.long),
			"output_ids": torch.tensor(input_ids_all, dtype=torch.long),
		}


def load_3class(**kwargs) -> []:
	assert kwargs.get("address") is not None
	with open(kwargs["address"], "r", encoding="utf-8") as f:
		all = []
		for content in f.read().split("\n"):
			text, label = content.split("$,$")
			if len(text) + len(kwargs["label_ids"]) < kwargs["max_seq_len"]:
				all.append([
					kwargs["prompt"] + "".join(text.split(" ")).strip(),
					eval(label)
				])
		newall = [[], [], []]
		for i in all:
			newall[i[1]].append(i)

		new_all = []
		for i in range(len(newall)):
			new_all += random.choices(newall[i], k=min([len(i) for i in newall]))
		random.shuffle(new_all)
		return new_all


def load_3labeled(dataset: BertDataset, address: str, label2id: [], max_len: int):
	with open(address, "r", encoding="utf-8") as f:
		templist = [i.split("$,$") for i in f.read().split("\n")]

	# dataset.data
	for i in templist:
		# [text, label]
		if 7 + len(i[0]) < max_len:
			dataset.data.append(
				f"[unused1][unused2][unused3]{label2id[eval(i[1])]}[unused4][unused5][unused6]" + i[0].strip()
			)


class BertTrainer:
	def __init__(self, args: Args):
		self.model = BertForMaskedLM.from_pretrained(args.model_path)
		self.args = args
		self.optimizer = self.build_optimizer(self.model)

	def build_optimizer(self, model):
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			 'weight_decay': self.args.weight_decay},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			 'weight_decay': 0.0}
		]

		# optimizer = AdamW(model.parameters(), lr=learning_rate)
		optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
		return optimizer

	def random_mask(self, text_ids: torch.Tensor):
		if self.args.tokenizer.mask_token_id in text_ids:
			return text_ids

		random_num = random.random()
		if random_num <= 0.8:
			text_ids[random.randint(0, len(text_ids))] = self.args.tokenizer.mask_token_id
			return text_ids
		elif random_num <= 0.9:
			random_select = random.randint(0, len(text_ids))
			text_ids[random_select + 1], text_ids[random_select] = text_ids[random_select], text_ids[random_select + 1]
			return text_ids
		else:
			return text_ids

	def train(self, train_data: DataLoader, test_data: DataLoader, dev_data: DataLoader):
		"""

		:param test_data:
		:param train_data: {
			"input_ids": torch.tensor, if the sentence is labeled which has been masked, else use random mask
			"attention_mask": torch.tensor,
			"token_type_ids": torch.tensor
			"output_ids": torch.tensor, ground Truth
		}
		:return:
		"""
		global_step_count = 0
		self.model.train()
		self.args.total_step = len(train_data)
		for epoch in range(1, self.args.epochs + 1):
			for batch_data in train_data:
				labeled = torch.where(batch_data["input_ids"] == self.args.tokenizer.mask_token_id,
				                      batch_data["output_ids"].to(self.args.device), -100)

				result = self.model(
					input_ids=batch_data["input_ids"].clone().apply_(self.random_mask).to(self.args.device),
					attention_mask=batch_data["attention_mask"].to(self.args.device),
					token_type_ids=batch_data["token_type_ids"].to(self.args.device),
				)

				criterion = torch.nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.08)
				labeled_loss = criterion(
					result.logits.view(-1, self.args.tokenizer.vocab_size)[self.args.label_ids],
					labeled.sort().indices)

				criterion = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.05)
				unlabeled_loss = criterion(
					result.logits.view(-1, self.args.tokenizer.vocab_size),
					batch_data["output_ids"].view(-1).to(self.args.device)
				)

				total_loss = labeled_loss + unlabeled_loss
				total_loss.backward()
				self.optimizer.step()
				global_step_count += 1
				pprint(f"[ Train ] Step: {global_step_count} / {self.args.total_step} Loss: {total_loss.item()}")
				if global_step_count % int(self.args.total_step * self.args.eval_epochs) == 0:
					self.test(test_data)
			self.test(test_data)
		self.dev(dev_data)

	def test(self, test_data: DataLoader):
		prediction, total_loss, labeled = self._test(test_data)
		acc = accuracy_score(labeled, prediction)
		pprint(f"[ TEST ] ACC: {acc}, Total loss: {total_loss}")
		if acc > self.args.best_acc:
			torch.save(
				self.model.state_dict(),
				self.args.ckpt_path
			)
			self.args.best_acc = acc
			with open(self.args.ckpt_path.split(".")[0] + ".txt", "w", encoding="utf-8") as f:
				f.write(self.args.toString())

	def _test(self, test_data: DataLoader):
		self.model.eval()
		total_loss = 0.
		prediction = []
		with torch.no_grad:
			for batch_data in test_data:
				# all in batch_data was labeled
				result = self.model(
					input_ids=batch_data["input_ids"].to(self.args.device),
					token_type_ids=batch_data["token_type_ids"].to(self.args.device),
					attention_mask=batch_data["attention_mask"].to(self.args.device)
				)

				labeled = torch.where(batch_data["input_ids"] == self.args.tokenizer.mask_token_id,
				                      batch_data["output_ids"].to(self.args.device), -100)

				criterion = torch.nn.CrossEntropyLoss(reduction="sum")
				total_loss += criterion(
					result.logits.view(-1, self.args.tokenizer.vocab_size)[self.args.label_ids],
					labeled.sort().indices
				)
				prediction += result.logits[batch_data["input_ids"] == self.args.tokenizer.mask_token_id][:, self.args.label_ids].argmax(-1).tolist()
		return prediction, total_loss, labeled.tolist().remove(-100)

	def dev(self, dev_data: DataLoader):
		prediction, total_loss, labeled = self._test(dev_data)
		acc = accuracy_score(labeled, prediction)
		pprint(f"[ DEV ] ACC: {acc}, Total loss: {total_loss}")
		pprint(f"[ DEV ] \n {classification_report(labeled, prediction, target_names=self.args.label2id.keys())}")
