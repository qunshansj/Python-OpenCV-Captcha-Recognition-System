python

class CaptchaGenerator:
    def __init__(self, captcha_array, captcha_size):
        self.captcha_array = captcha_array
        self.captcha_size = captcha_size

    def generate_captcha(self):
        image = ImageCaptcha()
        image_text = "".join(random.sample(self.captcha_array, self.captcha_size))
        image_path = r".\dataset\test\{}_{}.png".format(image_text, int(time.time()))
        image.write(image_text, image_path)

if __name__ == "__main__":
    captcha_array = list("123456789abcdefghijklmnopqrstuvwxyz")
    captcha_size = 6
    generator = CaptchaGenerator(captcha_array, captcha_size)
    for i in range(100):
        generator.generate_captcha()
