#!/bin/bash

declare file_name;
declare download_link;

if [ "$1" == "table_wise" ];
then
  file_name="rutabert_table_wise.tar.gz";
  download_link="https://drive.usercontent.google.com/download?id=1xLaXB9ZJA9G5JpiDP15D8NynzDX1ifWG&export=download&authuser=0&confirm=t&uuid=a35f908b-1469-463f-a6a0-5b9e538ee1c1&at=APZUnTXz_pe1JrgFTieEVijXeEQE:1712834480569";
elif [ "$1" == "column_wise" ]; then
  file_name="rutabert_column_wise.tar.gz";
  download_link="https://drive.usercontent.google.com/download?id=1yau0IrsRZZdeM0XVfRaN7p6SELgpIpAa&export=download&authuser=0&confirm=t&uuid=d5c6f6f8-da5b-4bc2-bb25-240fa34abd60&at=APZUnTVef6-2yBU3Ukk0oq-e-iUn:1712834703890";
else echo "Undefined download parameter."; exit;
fi

echo Downloading $file_name...;

# Download models
curl "$download_link" > $file_name;
mv $file_name ./checkpoints;
cd checkpoints && tar -xvf $file_name;
