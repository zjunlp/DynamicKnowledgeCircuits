from torch.utils.data import Dataset, DataLoader


def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, labels


class EAPDataset(Dataset):
    def __init__(self, data: list, task=None):
        self.data = data
        self.task = task
        self.task_template = {
            "birthday": "{subject} was born on",
            "city": "{subject} lives in the city of",
            "major": "{subject} majors in the field of",
            "university": "{subject} graduates from the",
            "company": "{subject} works for the company of",
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        clean_subject = data_point["clean_subject"]
        corrupted_subject = data_point["corrupted_subject"]
        clean = self.task_template[self.task].format(subject=clean_subject)
        corrupted = self.task_template[self.task].format(subject=corrupted_subject)
        label = [data_point["clean_label_idx"], data_point["corrupted_label_idx"]]
        return clean, corrupted, label

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
