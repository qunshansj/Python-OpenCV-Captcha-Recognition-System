python


class ImageProcessor:
    def __init__(self, k=4):
        self.k = k

    def remove_noise(self, img2):
        img2 = img2.convert('L')
        w, h = img2.size

        def get_neighbors(img, r, c):
            count = 0
            for i in [r - 1, r, r + 1]:
                for j in [c - 1, c, c + 1]:
                    if img.getpixel((i, j)) > 220:
                        count += 1
            return count

        for x in range(w):
            for y in range(h):
                if x == 0 or y == 0 or x == w - 1 or y == h - 1:
                    img2.putpixel((x, y), 255)
                else:
                    n = get_neighbors(img2, x, y)
                    if n > self.k:
                        img2.putpixel((x, y), 255)
        return img2

