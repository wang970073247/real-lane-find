
import cv2


def test():
    lena = cv2.imread("test_images/test1.jpg")
    while True:
        cv2.imshow('image', lena)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(cv2.waitKey(1) & 0xFF)
            print(ord('q'))
            print ("I'm done")
            #print(temp)
            break


if __name__ == '__main__':
    test()


