# A MapReduce implementation of the K-Means Algorithm in Spark

## Introduction
In this repository you'll find a MapReduce implementation of the K-Means algorithm. The code is written in Python since it has been developed for the Spark framework.

## Execute the code
For executing the code in the cluster you must run the following command:
```sh
spark-submit --master yarn ./main.py <input file> <centroids file> <cluster number> <output folder> <max iterations> <min centroids shift>
```
The arguments are mandatory in order to start the computations. 

## Documentation
The documentation of the didactic project related to this repository is available [here](https://github.com/fontanellileonardo/Spark-Kmeans/blob/master/doc/CLOUD_Project_Hadoop___Spark_Documentation.pdf).
In the documentation it is also reported the results of the **Hadoop** framework execution, the related repository is available [here](https://github.com/fontanellileonardo/Hadoop-Kmeans).

## Credits
D.Comola, E. Petrangeli, G. Alvaro, L. Fontanelli
