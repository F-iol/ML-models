Using fastapi there is a inference on localhost/docs, 
after subbmiting image of butterfly you should get prediction on site,
aswell as 3 top 'predictions' in terminal you ran docker image to check how confident model is in his prediction

build with 
docker built -t [name] .

run with
docker run --rm -p 8000:8000 [name]

