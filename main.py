import sys
import math
import pyspark
import numpy as np
import pathlib
import shutil


def parse_point(line):
    return np.array([float(coords) for coords in line.split()])


def compute_closest(point, cent):
    index = -1
    min_dist = math.inf
    for j in range(len(cent)):
        temp_dist = np.sqrt(np.sum((point - cent[j]) ** 2))
        if temp_dist < min_dist:
            min_dist = temp_dist
            index = j
	# in this key-value pair index is the key and (1, point) is the value
    return index, (1, point)


def delete_previous_output(output, sc):
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem

    fs = FileSystem.get(sc._jsc.hadoopConfiguration())
    fs.delete(Path(output))


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: ./main.py <inputFile> <centroidFile> <K> <outputFolder> <maxIterations> <minShift>")
        sys.exit()

    input_file = sys.argv[1]
    centroid_file = sys.argv[2]
    k = int(sys.argv[3])
    output = sys.argv[4]
    maxIterations = int(sys.argv[5])
    minShift = float(sys.argv[6])
    

    # initialization of a new Spark Context
    sc = pyspark.SparkContext(appName="KMeans", master="yarn")
    sc.setLogLevel("ERROR")

	# deletion of previous output 
    delete_previous_output(output, sc)

    # points is an RDD containing the points taken from input_file
    points = sc.textFile(input_file).map(parse_point).persist(pyspark.StorageLevel.MEMORY_AND_DISK)
	# initial_centroids is an array containing all the elements of the RDD that contained the centroids taken from centroid_file
    initial_centroids = sc.textFile(centroid_file).map(parse_point).collect()
    # print("Initial centroids:", initial_centroids)

    shift = math.inf
    iteration = 0

    while iteration < maxIterations and shift > minShift:
        print("Iteration:", iteration)
        # used a broadcast variable to save centroids values visible on all the worker nodes
        broadcast_centroids = sc.broadcast(initial_centroids)
		# mapping each point to the closest centroid
        closest_centroid = points.map(lambda x: (compute_closest(x, broadcast_centroids.value)))
		# sum of the occurrences (0) and of the coordinates (1) of each point reduced by key
        summed_points = closest_centroid.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        # new_centroids is an ordered array (by key) containing the coordinates of new centroids 
        new_centroids = summed_points.map(lambda v: (v[0], v[1][1] / v[1][0])).values().collect()
        print("New centroids:", new_centroids)

        shift = 0.0
		# computation of the euclidian distance between old and new centroids
        for c in range(len(new_centroids)):
            shift += np.sqrt(np.sum((new_centroids[c] - initial_centroids[c]) ** 2))
        print("Shift:", shift)

        initial_centroids = new_centroids
        new_centroids = []
        iteration += 1

    to_output = sc.parallelize(initial_centroids)
	# saving the final result in a text file
    to_output.saveAsTextFile(output)
  