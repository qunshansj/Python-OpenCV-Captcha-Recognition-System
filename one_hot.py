python

class CaptchaConverter:
    def __init__(self):
        self.captcha_size = common.captcha_size
        self.captcha_array = common.captcha_array

    # 字母转one-hot编码，如aabb转one-hot编码，词典为common.captcha_array
    def text2vc(self, text):
        vec = torch.zeros(self.captcha_size, len(self.captcha_array))
        for i in range(len(text)):
            vec[i, self.captcha_array.index(text[i])] = 1
        return vec

    # 还原，one-hot编码转字母
    def vecText(self, vec):
        vec = torch.argmax(vec, dim=1)
        text = ''
        for i in vec:
            text += self.captcha_array[i]
        return text

