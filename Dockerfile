
FROM ubuntu:24.04 AS build
RUN apt-get update -y
COPY . /webtool/
RUN apt-get install -y npm
WORKDIR /webtool/frontend/
RUN npm install
RUN npm run build
RUN apt-get install -y cmake g++
WORKDIR /webtool/backend/
RUN cmake -B build -S .
RUN cmake --build build
RUN cmake --install build --prefix /webtool/backend/package --component Runtime


FROM ubuntu:24.04
COPY --from=build /webtool/backend/package /usr/local/
COPY --from=build /webtool/frontend/dist/ /var/www/html/webtool
RUN ldconfig
VOLUME /var/www/html/webtool
EXPOSE 8080/tcp
CMD ["backend"]

