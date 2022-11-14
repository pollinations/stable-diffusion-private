cog build -t 614871946825.dkr.ecr.us-east-1.amazonaws.com/pollinations/stable-diffusion-private
 aws ecr get-login-password \
        --region us-east-1 \
    | docker login \
        --username AWS \
        --password-stdin 614871946825.dkr.ecr.us-east-1.amazonaws.com
docker push 614871946825.dkr.ecr.us-east-1.amazonaws.com/pollinations/stable-diffusion-private
docker inspect 614871946825.dkr.ecr.us-east-1.amazonaws.com/pollinations/stable-diffusion-private > inspect.json
cp meta.json ../model-index/pollinations/stable-diffusion-private/meta.json
cp inspect.json ../model-index/pollinations/stable-diffusion-private/inspect.json
cd ../model-index && python add_image.py stable-diffusion-private 614871946825.dkr.ecr.us-east-1.amazonaws.com/pollinations/stable-diffusion-private && cd ..