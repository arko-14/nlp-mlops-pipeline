FROM grafana/grafana:10.4.0
ENV GF_SECURITY_ALLOW_EMBEDDING=true
ENV GF_AUTH_ANONYMOUS_ENABLED=false
CMD ["/bin/bash","-lc","/run.sh &> /dev/null & sleep 3; /usr/sbin/grafana-server --homepath=/usr/share/grafana --http-port=${PORT} --packaging=docker"]
