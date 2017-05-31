# To start up model server:
#
# bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=lm --model_base_path=/tmp/serve/house-model-dynamic
#
# TODO(kjchavez): Add this to the start script in a sane way.

export FLASK_APP=server.py
export FLASK_DEBUG=1
flask run --host=0.0.0.0
