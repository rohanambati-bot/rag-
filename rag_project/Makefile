.PHONY: install ui api demo eval cicd docker-up docker-down clean

install:
	pip install -r requirements.txt

ui:
	streamlit run app.py

api:
	uvicorn api:app --reload --port 8000

demo:
	python main.py --mode demo

eval:
	python main.py --mode eval

cicd:
	python main.py --mode cicd --samples 50

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf outputs/plots/*.png outputs/*.csv outputs/*.md outputs/*.json
	rm -rf __pycache__ */__pycache__
