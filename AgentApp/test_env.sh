#!/bin/bash

ROOT="/home/jason/Auto-Image-Restoration"

# Defocus deblurring
# 1. DRBNet
PORT=8002

source activate drbnet

cd ${ROOT}/AgentApp/model_service/defocus_deblurring/DRBNet
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for DRBNet service start..."
  sleep 10
done

echo "Start testing DRBNet..."

sh client_test.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 2. IFAN
PORT=8003

source activate ifan
cd ${ROOT}/AgentApp/model_service/defocus_deblurring/IFAN
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for IFAN service start..."
  sleep 10
done

echo "Start testing IFAN..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 3. Restormer
PORT=8004

source activate restormer
cd ${ROOT}/AgentApp/model_service/defocus_deblurring/Restormer
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for Restormer service start..."
  sleep 10
done

echo "Start testing Restormer..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# Dehazing
# 1. DehazeFormer
PORT=8005

source activate dehazeformer
cd ${ROOT}/AgentApp/model_service/dehazing/DehazeFormer
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for DehazeFormer service start..."
  sleep 10
done

echo "Start testing Dehazing..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 2. RIDCP_dehazing
PORT=8006

source activate ridcp
cd ${ROOT}/AgentApp/model_service/dehazing/RIDCP_dehazing
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for RIDCP service start..."
  sleep 10
done

echo "Start testing RIDCP..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 3. X-Restormer
PORT=8007

source activate xrestormer
cd ${ROOT}/AgentApp/model_service/dehazing/X-Restormer
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for X-Restormer service start..."
  sleep 10
done

echo "Start testing X-Restormer..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 4. maxim
PORT=8008

source activate maxim
cd ${ROOT}/AgentApp/model_service/dehazing/maxim
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for maxim service start..."
  sleep 10
done

echo "Start testing maxim..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# denoising
# 1. MPRNet
PORT=8012

source activate mprnet
cd ${ROOT}/AgentApp/model_service/denoising/MPRNet
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for MPRNet service start..."
  sleep 10
done

echo "Start testing MPRNet..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 2. Restormer
PORT=8009

source activate restormer
cd ${ROOT}/AgentApp/model_service/denoising/Restormer
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for Restormer service start..."
  sleep 10
done

echo "Start testing Restormer..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 3. SwinIR
PORT=8013

source activate swinir
cd ${ROOT}/AgentApp/model_service/denoising/SwinIR
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for SwinIR service start..."
  sleep 10
done

echo "Start testing SwinIR..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 3. X-Restormer
PORT=8010

source activate xrestormer
cd ${ROOT}/AgentApp/model_service/denoising/X-Restormer
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for X-Restormer service start..."
  sleep 10
done

echo "Start testing X-Restormer..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 4. maxim
PORT=8011

source activate maxim
cd ${ROOT}/AgentApp/model_service/denoising/maxim
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for maxim service start..."
  sleep 10
done

echo "Start testing maxim..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# Deraining
# 1. MPRNet
PORT=8014

source activate mprnet
cd ${ROOT}/AgentApp/model_service/deraining/MPRNet
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for MPRNet service start..."
  sleep 10
done

echo "Start testing MPRNet..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 2. Restormer
PORT=8015

source activate restormer
cd ${ROOT}/AgentApp/model_service/deraining/Restormer
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for Restormer service start..."
  sleep 10
done

echo "Start testing Restormer..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 3. X-Restormer
PORT=8016

source activate xrestormer
cd ${ROOT}/AgentApp/model_service/deraining/X-Restormer
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for X-Restormer service start..."
  sleep 10
done

echo "Start testing X-Restormer..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 4. maxim
PORT=8017

source activate maxim
cd ${ROOT}/AgentApp/model_service/deraining/maxim
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for maxim service start..."
  sleep 10
done

echo "Start testing maxim..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# jpeg_compression_artifact_removal
# 1. FBCNN
PORT=8019

source activate fbcnn
cd ${ROOT}/AgentApp/model_service/jpeg_compression_artifact_removal/FBCNN
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for FBCNN service start..."
  sleep 10
done

echo "Start testing FBCNN..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 2. SwinIR
PORT=8018

source activate swinir
cd ${ROOT}/AgentApp/model_service/jpeg_compression_artifact_removal/SwinIR
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for SwinIR service start..."
  sleep 10
done

echo "Start testing SwinIR..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# Motion Deblur
# 1. MPRNet
PORT=8020

source activate mprnet
cd ${ROOT}/AgentApp/model_service/motion_deblurring/MPRNet
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for MPRNet service start..."
  sleep 10
done

echo "Start testing MPRNet..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 2. Restormer
PORT=8021

source activate restormer
cd ${ROOT}/AgentApp/model_service/motion_deblurring/Restormer
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for Restormer service start..."
  sleep 10
done

echo "Start testing Restormer..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 3. X-Restormer
PORT=8022

source activate xrestormer
cd ${ROOT}/AgentApp/model_service/motion_deblurring/X-Restormer
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for X-Restormer service start..."
  sleep 10
done

echo "Start testing X-Restormer..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 4. maxim
PORT=8023

source activate maxim
cd ${ROOT}/AgentApp/model_service/motion_deblurring/maxim
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for maxim service start..."
  sleep 10
done

echo "Start testing maxim..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# Super resolution
# 1. diffbir
PORT=8026

source activate diffbir
cd ${ROOT}/AgentApp/model_service/super_resolution/DiffBIR
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for diffbir service start..."
  sleep 10
done

echo "Start testing DiffBIR..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 2. HAT
PORT=8027

source activate hat
cd ${ROOT}/AgentApp/model_service/super_resolution/HAT
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for HAT service start..."
  sleep 10
done

echo "Start testing HAT..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 3. SwinIR
PORT=8024

source activate swinir
cd ${ROOT}/AgentApp/model_service/super_resolution/SwinIR
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for SwinIR service start..."
  sleep 10
done

echo "Start testing SwinIR..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo

# 4. X-Restormer
PORT=8025

source activate xrestormer
cd ${ROOT}/AgentApp/model_service/super_resolution/X-Restormer
nohup python model_serving.py >> log_serving.log 2>&1 &
SERVICE_PID=$!

while ! nc -z localhost $PORT; do
  echo "Waiting for X-Restormer service start..."
  sleep 10
done

echo "Start testing X-Restormer..."

sh client.sh

kill $SERVICE_PID
echo "Service with PID $SERVICE_PID stopped."
echo












