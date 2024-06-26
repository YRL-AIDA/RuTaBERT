#!/bin/bash

declare file_name;
declare download_link;

if [ "$1" == "table_wise" ];
then
  file_name="rutabert_table_wise.tar.gz";
  download_link="https://drive.usercontent.google.com/download?id=1HVKSVu_3kpHsu9HryTfAP-uFGlkXuCSQ&export=download&authuser=0&confirm=t&uuid=059405ea-89a4-4d71-bbf5-32bfc85131c5&at=APZUnTUgSTSwYijRC-6AC4j1ixHh%3A1719385959038";
elif [ "$1" == "column_wise" ]; then
  file_name="rutabert_column_wise.tar.gz";
  download_link="https://drive.usercontent.google.com/download?id=1QXXnmnAhmsmBbn-XxNZHm4WcWMQVJ1w7&export=download&authuser=0&confirm=t&uuid=c196ca4d-b679-4c0a-8374-d00c9926202e&at=APZUnTVnODqRMAwrZsvnOZZeD4-_:1719386096296";
else echo "Undefined download parameter."; exit;
fi

echo Downloading $file_name...;

# Download models
curl "$download_link" > $file_name;
mv $file_name ./checkpoints;
#cd checkpoints && tar -xvf $file_name;
