Все, кроме клиента, собирается docker-compose.
Клиент (папка producer) собирается и запускается:
sudo docker build -t producer .
sudo docker run -it --net task1_default producer
