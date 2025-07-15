# Use the official AWS base image for Python 3.11 Lambda
FROM public.ecr.aws/lambda/python:3.11

# Copy the requirements file into the image
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies using pip
# --no-cache-dir reduces image size slightly
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy the function code (api.py) into the image
# Ensure api.py is in the same directory as this Dockerfile (fraud_api_lambda/)
COPY api.py ${LAMBDA_TASK_ROOT}/

# Set the command that Lambda will invoke
# This should be the <filename>.<handler_variable_name> from template's original Handler property
CMD [ "api.handler" ]