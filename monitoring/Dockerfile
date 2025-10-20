FROM prom/prometheus:v2.55.0
RUN apk add --no-cache bash
ENV APP_HOST_PORT=app-xxxxx.onrender.com
COPY prometheus.render.yml /etc/prometheus/prometheus.tpl.yml
CMD ["/bin/bash","-lc","sed \"s#\\${APP_HOST_PORT}#${APP_HOST_PORT}#g\" /etc/prometheus/prometheus.tpl.yml > /etc/prometheus/prometheus.yml && /bin/prometheus --config.file=/etc/prometheus/prometheus.yml --web.listen-address=:10000"]
