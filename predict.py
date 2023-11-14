python

class CaptchaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path

    def predict_single(self, image_path):
        image = PIL.Image.open(image_path)
        image = noise_getout.remove_noise(image, k=4)
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((36, 170)),
        ])
        img_tensor = trans(image)
        img_tensor = img_tensor.reshape((1, 1, 36, 170))
        m = torch.load(self.model_path)
        output = m(img_tensor)
        output = output.view(-1, common.captcha_array.__len__())
        output_label = one_hot.vecText(output)
        return output_label

    def predict_group(self, path):
        test_dataset = my_dataset(path)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        test_len = test_dataset.__len__()
        correct = 0
        for i, (images, labels) in enumerate(test_dataloader):
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.view(-1, common.captcha_array.__len__())
            label_text = one_hot.vecText(labels)
            m = torch.load(self.model_path)
            output = m(images)
            output = output.view(-1, common.captcha_array.__len__())
            output_test = one_hot.vecText(output)
            if label_text == output_test:
                correct += 1
                print('正确值：{}，预测值：{}'.format(label_text, output_test))
            else:
                print('正确值{}，预测值{}'.format(label_text, output_test))
        print("测试样本总数{},预测正确率：{}".format(len(os.listdir(path)),
                                     correct / len(os.listdir(path)) * 100))

