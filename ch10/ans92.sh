# https://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392
export IP="0.0.0.0"
export PORT=5000
export URL_ROOT="/translator"
export CONFIG="./available_models/conf.json"
export HOST="127.0.0.1"

# NOTE that these parameters are optionnal
# here, we explicitely set to default values
python server.py --ip $IP --port $PORT --url_root $URL_ROOT --config $CONFIG

# curl http://$HOST:$PORT$URL_ROOT/models

# curl -i -X POST -H "Content-Type: application/json" \
#     -d '[{"src": "本日 は 晴天 なり", "id": 100}]' \
#     http://$HOST:$PORT$URL_ROOT/translate | perl -Xpne 's/\\u([0-9a-fA-F]{4})/chr(hex($1))/eg'
