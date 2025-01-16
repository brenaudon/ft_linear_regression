PROJECT_NAME=ft_linear_regression

pydoctor \
    --project-name=${PROJECT_NAME} \
    --project-version=0.1 \
    --project-url=https://github.com/brenaudon/$%7BPROJECT_NAME%7D/ \
    --html-base-url=https://brenaudon.github.io/$%7BPROJECT_NAME%7D/ \
    --docformat=epytext \
    *.py