#!/usr/bin/env bash


lib_dir="home/fungi/lib/python3.10/site-packages/pyspark/jars"
mkdir -p "${lib_dir}"
cd "${lib_dir}"
echo "Installing to $lib_dir."

wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar