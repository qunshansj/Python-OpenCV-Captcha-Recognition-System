python


class MyDataset(Dataset):
    def __init__(self, root_dir):
        super(MyDataset, self).__init__()
        self.image_path = [os.path.join(root_dir, image_name) for image_name in os.listdir(root_dir)]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((36, 170)),
        ])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = noise_getout.remove_noise(Image.open(image_path), k=4)
        image = self.transforms(image)
        label = image_path.split('\\')[-1]
        label = label.split('.')[0].casefold()
        label_tensor = one_hot.text2vc(label)
        label_tensor = label_tensor.view(1, -1)[0]
        return image, label_tensor

