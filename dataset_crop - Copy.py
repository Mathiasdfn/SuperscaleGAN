import tensorflow_datasets as tfds
import numpy as np

def dataset_crop(setName, setType):
   
    # Get tf dataset
    (trainingdata), metadata = tfds.load(setName, split=setType, with_info=True,
    as_supervised=True)

    
    # Iterate through it once for counting dataset size
    count = iter(trainingdata)
    total_number_img = sum(1 for _ in count)
    
    # Reset iterator for cropping images
    the_set = iter(trainingdata)
    
    # Initiate the new dataset and setup a dummy image for np.vstack
    dataset = np.zeros((1,100,100,3))
    
    # Iterate through dataset
    for index, i in enumerate(the_set):
        image = (i[0].numpy()).astype('int')
        image = np.expand_dims(image,0)
        
        # Count how many sections of 100 pixels on the x and y axis
        img_y_count = image.shape[1] // 100
        img_x_count = image.shape[2] // 100
        
        # cutting up the images and add it to the new dataset
        for y in (np.arange(img_y_count) + 1):
            for x in (np.arange(img_x_count) + 1):
                dataset = np.vstack((dataset, image[:, y*100-100:y*100, x*100-100:x*100, :]))
        
        #print progress evert 20 images
        if (index + 1) % 20 == 0:
            print("Picture number {} out of {} done".format((index+1), total_number_img))
        
        # Remove comments to test that the code is working properly
        #if (index + 1) == 10:
        #   break
    
    # Remove dummy image from the new dataset
    dataset = dataset[1:,...] 
    
    np.savez("{}.npz".format((setName + "_" + setType)), dataset = dataset)

dataset_crop('oxford_flowers102', "train")
dataset_crop('oxford_flowers102', "test")
dataset_crop('oxford_flowers102', "validation")

# Note that when loading the images into a script, the arraytype has to be 'int'
# for the images to render properly with imshow