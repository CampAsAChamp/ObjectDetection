import numpy as np
import matplotlib.pyplot as plt

# we will only use:  scipy.ndimage.correlate
from scipy import ndimage

# we will only use:  ski.util.view_as_windows for computing hog descriptor
import skimage as ski
# we will only use:  scipy.ndimage.correlate
from scipy import ndimage
import math



def mygradient(image):
    """
    This function takes a grayscale image and returns two arrays of the
    same size, one containing the magnitude of the gradient, the second
    containing the orientation of the gradient.


    Parameters
    ----------
    image : 2D float array of shape HxW
         An array containing pixel brightness values

    Returns
    -------
    mag : 2D float array of shape HxW
        gradient magnitudes

    ori : 2Dfloat array of shape HxW
        gradient orientations in radians
    """

    mag = np.zeros(image.shape)
    ori = np.zeros(image.shape)

    xDerivative = ndimage.correlate(image, np.array([[-1, 1]]), mode="nearest")
    yDerivative = ndimage.correlate(image, np.array([[-1], [1]]), mode="nearest")

    xSquared = np.multiply(xDerivative, xDerivative)
    ySquared = np.multiply(yDerivative, yDerivative)

    mag = np.sqrt(np.add(xSquared, ySquared))

    # Floating Point correction
    xDerivative = np.add(xDerivative, 0.0000000000000000001)
    ori = np.arctan(np.divide(yDerivative, xDerivative))

    return (mag, ori)




def hog(image, bsize, norient):
    """
    This function takes a grayscale image and returns a 3D array
    containing the histogram of gradient orientations descriptor (HOG)
    We follow the convention that the histogram covers gradients starting
    with the first bin at -pi/2 and the last bin ending at pi/2.

    Parameters
    ----------
    image : 2D float array of shape HxW
         An array containing pixel brightness values

    bsize : int
        The size of the spatial bins in pixels, defaults to 8

    norient : int
        The number of orientation histogram bins, defaults to 9

    Returns
    -------
    ohist : 3D float array of shape (H/bsize,W/bsize,norient)
        edge orientation histogram

    """

    # determine the size of the HOG descriptor
    (h, w) = image.shape
    h2 = int(np.ceil(h / float(bsize)))
    w2 = int(np.ceil(w / float(bsize)))
    ohist = np.zeros((h2, w2, norient))

    # pad the input image as needed so that it is a multiple of bsize
    # If the input image dimensions are not a multiple of 8,
    # you should use np.pad with the mode=edge option to pad the width and height up to the nearest integer multiple of 8.
    if h % bsize == 0:
        ph = h
    else:
        ph = bsize * (math.trunc((h + (bsize)) / bsize))

    if w % bsize == 0:
        pw = w
    else:
        pw = bsize * (math.trunc((w + (bsize)) / bsize))

    ph = ph - h
    pw = pw - w

    image = np.pad(image, (ph, pw), 'symmetric')

    # compute image gradients
    (magCopy, oriCopy) = mygradient(image)

    # choose a threshold which is 10% of the maximum gradient magnitude in the image
    thresh = 0.1 * np.max(magCopy)

    # separate out pixels into orientation channels, dividing the range of orientations
    # [-pi/2,pi/2] into norient equal sized bins and count how many fall in each block
    # as a sanity check, make sure every pixel gets assigned to at most 1 bin.

    for i in range(norient):
        # create a binary image containing 1s for pixels at the ith
        # orientation where the magnitude is above the threshold.

        # I would suggest your function loops over the orientation bins.
        # For each orientation bin you'll need to identify those pixels in the image
        # whose magnitude is above the threshold and whose orientation falls in the given bin.
        # You can do this easily in numpy using logical operations in order to generate an array
        # the same size as the image that contains Trues at the locations of every edge pixel
        # that falls in the given orientation bin and is above threshold

        B = np.zeros((h, w))  # same size as image

        binStart = (-np.pi / 2) + i * (np.pi / 9)
        binEnd = (-np.pi / 2) + (i + 1) * (np.pi / 9)

        mag = np.zeros(image.shape)
        mag[np.where(magCopy > thresh)] = 1

        ori = np.zeros(image.shape)
        ori[np.where((binStart <= oriCopy) & (oriCopy <= binEnd))] = 1

        B = mag * ori

        # pull out non-overlapping bsize x bsize blocks
        chblock = ski.util.view_as_windows(B, (bsize, bsize), step=bsize)

        # sum up the count for each block and store the results
        for j in range(h2):
            for k in range(w2):
                ohist[j, k, i] = np.count_nonzero(chblock[j, k])

    # lastly, normalize the histogram so that the sum along the orientation dimension is 1
    # note: don't divide by 0! If there are no edges in a block (i.e. the sum of counts
    # is 0) then your code should leave all the values as zero.
    mySum = np.sum(ohist, axis=2)

    for g in range(h2):
        for h in range(w2):
            if (mySum[g, h] > 0.0):
                ohist[g, h, :] = ohist[g, h, :] / mySum[g, h]

    #     print("ohist", ohist)
    assert (ohist.shape == (h2, w2, norient))

    return ohist


import scipy


def detect(image, template, ndetect=5, bsize=8, norient=9):
    """
    This function takes a grayscale image and a HOG template and
    returns a list of detections where each detection consists
    of a tuple containing the coordinates and score (x,y,score)

    Parameters
    ----------
    image : 2D float array of shape HxW
         An array containing pixel brightness values

    template : a 3D float array
        The HOG template we wish to match to the image

    ndetect : int
        Number of detections to return

    bsize : int
        The size of the spatial bins in pixels, defaults to 8

    norient : int
        The number of orientation histogram bins, defaults to 9

    Returns
    -------
    detections : a list of tuples of length ndetect
        Each detection is a tuple (x,y,score)

    """

    # norient for the template should match the norient parameter passed in
    assert (template.shape[2] == norient)

    fmap = hog(image, bsize, norient)

    # cross-correlate the template with the feature map to get the total response
    resp = np.zeros(fmap.shape)
    for i in range(norient):
        resp = resp + scipy.ndimage.correlate(fmap, template)  # i did this

    #     print("Resp:", resp)
    # sort the values in resp in descending order.
    # val[i] should be ith largest score in resp
    # ind[i] should be the index at which it occurred so that val[i]==resp[ind[i]]
    val = np.sort(resp.flatten())[::-1]  # sorted response values
    ind = np.dstack(np.unravel_index(np.argsort(resp.ravel()), (resp.shape)))  # corresponding indices
    ind = np.rot90(np.rot90(ind))

    # work down the list of responses from high to low, to generate a
    # list of ndetect top scoring matches which do not overlap
    detcount = 0
    i = 0
    detections = []
    while ((detcount < ndetect) and (i < len(val))):
        # convert 1d index into 2d index
        yb = ind[0][i][1]
        xb = ind[0][i][0]

        # assert(val[i]==resp[yb,xb]) #make sure we did indexing correctly

        # covert block index to pixel coordinates based on bsize
        xp = bsize * xb
        yp = bsize * yb

        # check if this detection overlaps any detections that we've already added
        # to the list. compare the x,y coordinates of this detection to the x,y
        # coordinates of the detections already in the list and see if any overlap
        # by checking if the distance between them is less than 70% of the template
        # width/height

        overlap = False
        for j in detections:
            dist = math.sqrt((j[0] - xp) ** 2 + (j[1] - yp) ** 2)
            if dist < (0.7 * template.shape[1] * bsize):  # (width/bsize) *bsize *0.7
                overlap = True

        # if the detection doesn't overlap then add it to the list
        if not overlap:
            detcount = detcount + 1
            detections.append((yp, xp, val[i]))

        i = i + 1

    if (len(detections) < ndetect):
        print('WARNING: unable to find ', ndetect, ' non-overlapping detections')

    return detections


import matplotlib.patches as patches


def plot_detections(image, detections, tsize_pix):
    """
    This is a utility function for visualization that takes an image and
    a list of detections and plots the detections overlayed on the image
    as boxes.

    Color of the bounding box is based on the order of the detection in
    the list, fading from green to red.

    Parameters
    ----------
    image : 2D float array of shape HxW
         An array containing pixel brightness values

    detections : a list of tuples of length ndetect
        Detections are tuples (x,y,score)

    tsize_pix : (int,int)
        The height and width of the box in pixels

    Returns
    -------
    None

    """
    ndetections = len(detections)

    fig = plt.figure()
    fig.add_subplot(1, 1, 1).imshow(image, cmap=plt.cm.gray)
    subplot = plt.gca()
    w = tsize_pix[1]
    h = tsize_pix[0]
    red = np.array([1, 0, 0])
    green = np.array([0, 1, 0])
    ct = 0
    for (x, y, score) in detections:
        xc = x - (w // 2)
        yc = y - (h // 2)
        col = (ct / ndetections) * red + (1 - (ct / ndetections)) * green
        rect = patches.Rectangle((xc, yc), w, h, linewidth=3, edgecolor=col, facecolor='none')
        subplot.add_patch(rect)
        ct = ct + 1

    plt.show()

    from skimage.transform import resize

def learn_template(posfiles, negfiles, tsize=np.array([16, 16]), bsize=8, norient=9):
    """
    This function takes a list of positive images that contain cropped
    examples of an object + negative files containing cropped background
    and a template size. It produces a HOG template and generates visualization
    of the examples and template

    Parameters
    ----------
    posfiles : list of str
         Image files containing cropped positive examples

    negfiles : list of str
        Image files containing cropped negative examples

    tsize : (int,int)
        The height and width of the template in blocks

    Returns
    -------
    template : float array of size tsize x norient
        The learned HOG template

    """

    # compute the template size in pixels
    # corresponding to the specified template size (given in blocks)
    tsize_pix = bsize * tsize

    # figure to show positive training examples
    fig1 = plt.figure()
    pltct = 1

    # accumulate average positive and negative templates
    pos_t = np.zeros((tsize[0], tsize[1], norient), dtype=float)
    first = True
    for file in posfiles:
        # load in a cropped positive example
        image = plt.imread(file)
        if (image.dtype == np.uint8):
            image = image.astype(float) / 256
        image = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3

        # convert to grayscale and resize to fixed dimension tsize_pix
        # using skimage.transform.resize if needed.
        img_scaled = resize(image, tsize_pix)  # might needa put tsize twice for hw

        # display the example if you want to train with a large # of examples,
        # you may want to modify this, e.g. to show only the first 5.
        ax = fig1.add_subplot(len(posfiles), 1, pltct).imshow(img_scaled, cmap=plt.cm.gray)
        pltct = pltct + 1

        # extract feature
        fmap = hog(img_scaled, bsize, norient)
        pos_t = np.add(fmap, pos_t)

    pos_t = (1 / len(posfiles)) * pos_t
    fig1.show()

    # repeat same process for negative examples
    fig2 = plt.figure()

    negct = 1
    neg_t = np.zeros((tsize[0], tsize[1], norient), dtype=float)
    first = True
    for file in negfiles:
        image = plt.imread(file)
        if (image.dtype == np.uint8):
            image = image.astype(float) / 256
        image = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3

        # convert to grayscale and resize to fixed dimension tsize_pix
        # using skimage.transform.resize if needed.
        img_scaled = resize(image, tsize_pix)  # might needa put tsize twice for hw

        ax = fig2.add_subplot(len(negfiles), 1, negct).imshow(img_scaled, cmap=plt.cm.gray)
        negct = negct + 1
        fmap = hog(img_scaled, bsize, norient)

        neg_t = np.add(fmap, neg_t)

    neg_t = (1 / len(negfiles)) * neg_t
    fig2.show()

    # add code here to visualize the positive and negative parts of the template
    # using hogvis. you should separately visualize pos_t and neg_t rather than
    # the final tempalte.

    posfig = plt.figure()
    h = hogvis.hogvis(pos_t, bsize, norient)
    posfig.add_subplot(1, 1, 1).imshow(h, cmap=plt.cm.gray)
    posfig.show()

    negfig = plt.figure()
    h1 = hogvis.hogvis(neg_t, bsize, norient)
    negfig.add_subplot(1, 1, 1).imshow(h1, cmap=plt.cm.gray)
    negfig.show()

    # now construct our template as the average positive minus average negative
    template = np.subtract(pos_t, neg_t)

    templatefig = plt.figure()
    h2 = hogvis.hogvis(template, bsize, norient)
    templatefig.add_subplot(1, 1, 1).imshow(h2, cmap=plt.cm.gray)
    templatefig.show()

    return template


#############################
##
#############################
#
# Demonstrate your mygradient function here by loading in a grayscale
# image, calling mygradient, and visualizing the resulting magnitude
# and orientation images. For visualizing orientation image, I suggest
# using the hsv or twilight colormap.
#

image = plt.imread("normal1.jpg")

if(image.dtype == np.uint8):
    image = image.astype(float) /256

image = (image[:,:,0]+image[:,:,1]+image[:,:,2])/3


(mag,ori) = mygradient(image)

plt.imshow(image, cmap= plt.cm.gray)
plt.show()

plt.imshow(mag, cmap=plt.cm.gray)
plt.show()

plt.imshow(ori, cmap=plt.cm.twilight)
plt.show()


###############################
##
###############################
#provided function for visualizing hog descriptors
import hogvis as hogvis

#
# generate a simple test image... a 80x80 image
# with a circle of radius 30 in the center
#
[yy,xx] = np.mgrid[-44:44,-44:44]
im = np.array((xx*xx+yy*yy<=30*30),dtype=float)

plt.imshow(im)
plt.show()

h = hog(im, 8, 9)

h = hogvis.hogvis(h,8,9)
plt.imshow(h)
plt.show()

################################
##
################################
#
# sketch of some simple test code, modify as needed
#

# create a synthetic image
[yy, xx] = np.mgrid[-40:40, -40:40]
im1 = np.array((xx * xx + yy * yy <= 30 * 30), dtype=float)
[yy, xx] = np.mgrid[-60:20, -60:20]
im2 = np.array((xx * xx + yy * yy <= 25 * 25), dtype=float)
im = 0.5 * im1 + 0.5 * im2

plt.imshow(im1)
plt.show()

plt.imshow(im2)
plt.show()

plt.imshow(im)
plt.show()

# compute feature map with default parameters
fmap = hog(im, 8, 9)

# extract a 3x3 template
template = fmap[1:4, 1:4, :]

# run the detect code
detections = detect(im, template, ndetect=5)

# visualize results.
plot_detections(im, detections, (24, 24))

# visually confirm that:
#   1. top detection should be the same as the location where we selected the template
#   2. multiple detections do not overlap too much

###################################
##
###################################
# assume template is 16x16 blocks, you may want to adjust this
# for objects of different size or aspect ratio.
# compute image a template size
bsize=8
tsize=np.array([16,16])  #height and width in blocks
tsize_pix = bsize*tsize  #height and width in pixels
posfiles = ('faces5.jpg', 'faces6.jpg', 'faces7.jpg', 'faces8.jpg')
negfiles = ('neg1.jpg', 'neg2.jpg', 'neg3.jpg', 'neg4.jpg', 'neg5.jpg')

# call learn_template to learn and visualize the template and training data
template=learn_template(posfiles,negfiles,tsize=tsize)


image = plt.imread('normal1.jpg')

if (image.dtype == np.uint8):
    image = image.astype(float) / 256

image = (image[:,:,0]+image[:,:,1]+image[:,:,2])/3
# image=resize(image, tsize_pix)

detections = detect(image,template, 5, 8, 9)
plot_detections(image,detections,tsize_pix)

###########################################

# assume template is 16x16 blocks, you may want to adjust this
# for objects of different size or aspect ratio.
# compute image a template size
bsize=8
tsize=np.array([16,16])  #height and width in blocks
tsize_pix = bsize*tsize  #height and width in pixels
posfiles = ('crop1.jpg', 'crop5.jpg')
negsignfiles = ('anotherneg4.jpg', 'negsign4.jpg')

# call learn_template to learn and visualize the template and training data
template=learn_template(posfiles,negsignfiles,tsize=tsize)


image = plt.imread('normalsign3.jpg')

if (image.dtype == np.uint8):
    image = image.astype(float) / 256

image = (image[:,:,0]+image[:,:,1]+image[:,:,2])/3
# image=resize(image, tsize_pix)

detections = detect(image,template, 5, 8, 9)
plot_detections(image,detections,tsize_pix)




