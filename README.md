# Ai-of-infrastruktur-miniprojekt
Source for code for the model and deployment script for FastApi and docker


Pre deployment models contain the source code before any implementation, and includes a Script_with_visualizations which was used to analyse each cluster. 

The app directory contains the neccesary script for deploying the model and also a directory for the models path. 

sample data to check if the models work. 

Change Request body to this for cluster 0 
{
  "order_hour": 14,
  "order_dayofweek": 2,
  "quantity": 3,
  "unit_price": 13,
  "total_price": 33
}

Change Request body to this for cluster 1 

{
  "order_hour": 22,
  "order_dayofweek": 7,
  "quantity": 3,
  "unit_price": 13,
  "total_price": 33
}
