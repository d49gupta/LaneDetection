import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os

def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g

def canny_edge_detection(image, low_threshold, high_threshold, kernel_size, sigma):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = cv.filter2D(gray, -1, kernel)

    # Compute gradient magnitude and direction
    dx = cv.Sobel(smoothed, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(smoothed, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(dx, dy)
    direction = np.arctan2(dy, dx)

    # Perform non-maximum suppression
    suppressed = np.zeros_like(magnitude)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q, r = magnitude[i, j + 1], magnitude[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q, r = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q, r = magnitude[i + 1, j], magnitude[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q, r = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]

    # Perform Double Thresholding
    edges = np.zeros_like(suppressed)
    weak = low_threshold
    strong = high_threshold

    strong_i, strong_j = np.where(suppressed >= strong)
    zeros_i, zeros_j = np.where(suppressed < low_threshold)
    weak_i, weak_j = np.where((suppressed <= high_threshold) & (suppressed >= low_threshold))

    edges[strong_i, strong_j] = strong
    edges[weak_i, weak_j] = weak

    # Perform Hysterisis 
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if edges[i, j] == weak:
                if (
                    (edges[i + 1, j - 1] == strong)
                    or (edges[i + 1, j] == strong)
                    or (edges[i + 1, j + 1] == strong)
                    or (edges[i, j - 1] == strong)
                    or (edges[i, j + 1] == strong)
                    or (edges[i - 1, j - 1] == strong)
                    or (edges[i - 1, j] == strong)
                    or (edges[i - 1, j + 1] == strong)
                ):
                    edges[i, j] = strong
                else:
                    edges[i, j] = 0
    return edges

def find_accuracy(canny_image, self_image):

    difference = np.zeros_like(canny_image)
    counter = 0

    for i in range(1, canny_image.shape[0] - 1):
        for j in range(1, canny_image.shape[1] - 1):
            # if canny_image[i, j] == self_image[i, j] == 255:
            #     counter = counter + 1
            difference[i, j] = abs(canny_image[i, j] - self_image[i, j])

        num_zeros = np.count_nonzero(difference == 0)
    return num_zeros*100/difference.size

def precision(ground_truth, canny_edges):
    true_positives = np.logical_and(ground_truth, canny_edges).sum()
    detected_positives = canny_edges.sum()
    return true_positives / detected_positives

def recall(ground_truth, canny_edges):
    true_positives = np.logical_and(ground_truth, canny_edges).sum()
    actual_positives = ground_truth.sum()
    return true_positives / actual_positives

def f1_score(ground_truth, canny_edges):
    prec = precision(ground_truth, canny_edges)
    rec = recall(ground_truth, canny_edges)
    return 2 * (prec * rec) / (prec + rec)

def mean_squared_error(ground_truth, canny_edges):
    return ((ground_truth.astype(float) - canny_edges.astype(float)) ** 2).mean()


if __name__ == "__main__":
    image = cv.imread(r'C:\Users\16134\OneDrive - University of Waterloo\Documents\School\Second Year\2B\MTE 203\Project 2\Images\23050.jpg')
    
    # Set the parameters for Canny edge detection, adjust appropriately 
    low_threshold = 125 #what defines irellevant pixels
    high_threshold = 175 #what defines strong pixels
    kernel_size = 7 #kernel window size
    sigma = 0.3 #standard deviation of the Gaussian filter (amount of smoothing)
    total_pixels = image.shape[0]*image.shape[1] #product of number of pixels in width and height

    edges_gray = canny_edge_detection(image, low_threshold, high_threshold, kernel_size, sigma)
    edges_canny = cv.Canny(image, low_threshold, high_threshold)


    groundtruth_file = r'C:\Users\16134\OneDrive - University of Waterloo\Documents\School\Second Year\2B\MTE 203\Project 2\23050.mat'
    groundtruth_folder = os.path.dirname(groundtruth_file)
    files = os.listdir(groundtruth_folder)
    groundtruth_file_path = os.path.join(groundtruth_folder, '23050.mat')
    mat_contents = sio.loadmat(groundtruth_file_path)
    ground_truth = mat_contents['groundTruth'][0, 0]['Boundaries'][0, 0]
    ground_truth_image = cv.resize(ground_truth, (edges_gray.shape[1], edges_gray.shape[0]))


    # # # Split the image into color channels
    # # b, g, r = cv.split(image)
    # # edges_b = canny_edge_detection(b, low_threshold, high_threshold, kernel_size, sigma)
    # # edges_g = canny_edge_detection(g, low_threshold, high_threshold, kernel_size, sigma)
    # # edges_r = canny_edge_detection(r, low_threshold, high_threshold, kernel_size, sigma)
    # # edges_combined = cv.bitwise_or(cv.bitwise_or(edges_b, edges_g), edges_r)
    # # num_edges_RGB = int((np.count_nonzero(edges_b) + np.count_nonzero(edges_g) + np.count_nonzero(edges_r)) / 3) #average edges of all 3 color channels
    # # print("Number of RGB edges:", num_edges_RGB, "and the percentage of edge pixels is:", num_edges_RGB/total_pixels*100, "%")
    # # cv.imshow("Color Channel Edges", edges_combined)

    num_edges_gray = np.count_nonzero(edges_gray)
    num_edges_canny = np.count_nonzero(edges_canny)

    # Results
    accuracy = find_accuracy(edges_canny, edges_gray)

    prec = precision(ground_truth_image, edges_gray)
    rec = recall(ground_truth_image, edges_gray)
    f1 = f1_score(ground_truth_image, edges_gray)
    mse = mean_squared_error(ground_truth_image, edges_gray)

    prec1 = precision(ground_truth_image, edges_canny)
    rec1 = recall(ground_truth_image, edges_canny)
    f2 = f1_score(ground_truth_image, edges_canny)
    mse1 = mean_squared_error(ground_truth_image, edges_canny)

    print("Total number of pixels: ", total_pixels)
    print("Number of gray edges:", num_edges_gray, "and the percentage of edge pixels is:", num_edges_gray/total_pixels*100, "%")
    print("Number of canny edges:", num_edges_canny, "and the percentage of edge pixels is:", num_edges_canny/total_pixels*100, "%")
    print("Me vs Canny:", accuracy)
    print("My Precision:", prec, "vs Canny's Prescision:", prec1) # ratio of detected edges to true edges
    print("My Recall:", rec, "vs Canny's Recall:", rec1) # percentage of detected true edges
    print("My F1 Score:", f1, "vs Canny's F1 Score:", f2) # 0 to 1 (best)
    print("My Mean Squared Error:", mse, "vs Canny's Mean Squared Error:", mse1) # large difference between detected edges and truth edges

    # Display the original image and the detected edges
    cv.imshow("Original Image", image)
    cv.imshow("My Canny Image", edges_gray)
    cv.imshow("Canny Image", edges_canny)
    cv.waitKey(0)
    cv.destroyAllWindows()
