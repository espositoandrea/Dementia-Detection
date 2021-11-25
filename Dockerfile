FROM python:3.9-slim AS base
WORKDIR /code

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1

FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install pipenv
RUN apt-get update && apt-get install -y gcc python3-opencv

# Install python dependencies in /.venv
COPY Pipfile .
#COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy

#FROM base as runtime
#COPY --from=python-deps /code/.venv /code/.venv
ENV PATH="/code/.venv/bin:$PATH"

COPY ./src /code/app
COPY ./data/model/memento /code/app/data/model/memento
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
