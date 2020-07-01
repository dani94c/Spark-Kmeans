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
    return index, point


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: ./main.py <inputFile> <K> <seed> <outputFile> <maxIterations> <minShift>")
        sys.exit()

    input_file = sys.argv[1]
    k = int(sys.argv[2])
    seed = int(sys.argv[3])
    output_file = sys.argv[4]
    maxIterations = int(sys.argv[5])
    minShift = float(sys.argv[6])

    output_path = pathlib.Path(output_file)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
        print("Removed output directory")

    # initialization of a new Spark Context
    sc = pyspark.SparkContext(appName="KMeans", master="yarn")

    points = sc.textFile(input_file).map(parse_point).persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    # print("Points of inputFile are:", points.collect())

    # takeSample(withReplacement, number, seed): prendo k centroidi random tra i punti in input
    initial_centroids = points.takeSample(False, k, seed)
    print("Initial centroids:", initial_centroids)

    shift = math.inf
    iteration = 0

    while iteration < maxIterations and shift > minShift:
        closest_centroid = points.map(lambda x: (compute_closest(x, initial_centroids)))
        # print("Closest centroid:", closest_centroid.collect())
        # new centroid is an ordered array containing the values of the new centroids from the smallest to biggest key
        new_centroids = closest_centroid.reduceByKey(lambda x, y: np.mean([x,y], axis=0)).sortByKey().values().collect()
        print("Iteration:", iteration, "Centroids:", new_centroids)

        shift = 0.0
        for c in range(len(initial_centroids)):
            shift += np.sqrt(np.sum((initial_centroids[c] - new_centroids[c]) ** 2))
        print("Iteration:", iteration, "shift:", shift)

        initial_centroids = new_centroids
        new_centroids = []
        iteration += 1

    to_output = sc.parallelize(initial_centroids)
    to_output.saveAsTextFile(output_file)
