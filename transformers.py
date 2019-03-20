from skimage import transform

class Resize(object):
    """
    Rescale the input image to a given width and height
    """

    def __init__(self, width=None, height=None):
        self.width = width
        self.height = height


    def __call__(self, image):
        #assumes image is of type numpy image
        image_height, image_width = image.shape[:2] #numpy image is stored as height, width and depth
        if image_width > self.width and image_height > self.height:
            image = transform.resize(image, (self.height, self.width))

        return image

class ToTensor(object):

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return image
