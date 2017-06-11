from tesserocr import PyTessBaseAPI

images = ['t1.jpg', 't2.jpg', 't3.jpg']

with PyTessBaseAPI() as api:
    for img in images:
        api.SetImageFile(img)
        print api.GetUTF8Text()
        print api.AllWordConfidences()
