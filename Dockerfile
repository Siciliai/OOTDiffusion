FROM python:3.10

RUN apt-get update && apt-get install -y libgl1

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' > /etc/timezone

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -i https://mirrors.aliyun.com/pypi/simple/

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . /app

WORKDIR /app/run

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "7866", "api:app"]