import { useEffect, useRef, useState } from "react";
import { notification } from "antd"
import { backendServer, processFetchResponse } from "../common/utils";
import s from "./MapPinning.module.css"

declare global {
  interface Window {
    BMapGL: {
      Map: any;
      Point: any;
      Marker: any;
    };
    displayOptions: any;
  }
}

function MapPinning() {
  const mapInstance = useRef<any>(null);
  const mapRef = useRef<HTMLDivElement | null>(null);
  const markersRef = useRef<any[]>([]);
  const deleteBtnRef = useRef<HTMLButtonElement | null>(null);
  const searchBtnRef = useRef<HTMLButtonElement | null>(null);

  const [titleText, setTitleText] = useState<string>("");
  const [markerPicked, setMarkerPicked] = useState<any>(null);
  const [_, forceRender] = useState(0);
  const [password, setPassword] = useState("");
  const [searchText, setSearchText] = useState("");

  const [notifyApi, notifyContext] = notification.useNotification();

  // 初始化地图
  useEffect(() => {
    const { BMapGL } = window;

    mapInstance.current = new BMapGL.Map(mapRef.current);
    mapInstance.current.centerAndZoom(new BMapGL.Point(116.404, 39.915), 15);
    mapInstance.current.enableScrollWheelZoom();
    mapInstance.current.disableContinuousZoom();

    mapInstance.current.addEventListener("click", (e: any) => {
      if (e.overlay != null) return;

      const marker = new BMapGL.Marker(new BMapGL.Point(e.latlng.lng, e.latlng.lat));
      marker.addEventListener("click", (_: any) => {
        setMarkerPicked(marker);
      });
      markersRef.current.push(marker);
      setMarkerPicked(marker);
      mapInstance.current.addOverlay(marker);
    });
  }, []);

  const deletePoint = () => {
    markersRef.current = markersRef.current.filter((marker) => marker != markerPicked);
    setMarkerPicked(markersRef.current[markersRef.current.length - 1]);
    mapInstance.current.removeOverlay(markerPicked);
  };

  const backend_server = backendServer();

  const upload = async () => {
    const body_content = markersRef.current.map((x) => {
      return { lat: x.latLng.lat, lng: x.latLng.lng, text: x.getTitle() }
    })
    try {
      const response = await fetch(`${backend_server}/api/map_pinning/upload`, {
        method: "POST",
        body: JSON.stringify({ name: titleText, password: password, markers: body_content }),
      });
      await processFetchResponse(response, notifyApi, '上传');
    } catch (error) {
      notifyApi.error({ message: '上传失败', duration: 3 });
      console.error(error);
    }
  };

  const download = async () => {
    try {
      const { BMapGL } = window;

      const response = await fetch(`${backend_server}/api/map_pinning/download?name=${titleText}`);
      if (!await processFetchResponse(response, notifyApi, '下载')) {
        return;
      }
      const { markers } = await response.json();
      markers.map((x: { lat: number, lng: number, text: string }) => {
        const marker = new BMapGL.Marker(new BMapGL.Point(x.lng, x.lat));
        marker.setTitle(x.text);
        marker.addEventListener("click", (_: any) => {
          setMarkerPicked(marker);
        });
        markersRef.current.push(marker);
        setMarkerPicked(marker);
        mapInstance.current.addOverlay(marker);
      });
    } catch (error) {
      notifyApi.error({ message: '下载失败', duration: 3 });
      console.error(error);
    }
  };

  const search = () => {
    mapInstance.current.centerAndZoom(searchText, 18);
  };

  return (
    <div className={s.root}>
      {notifyContext}
      <div className={s.infoContainer}>
        <input type="text" placeholder="地点集合" value={titleText}
          onChange={(e) => setTitleText(e.target.value)} />
        <button onClick={download}
          disabled={markersRef.current.length != 0 || titleText.length == 0}>
          下载
        </button>
        <input type="password" placeholder="密码" value={password}
          onChange={(e) => setPassword(e.target.value)} />
        <button onClick={upload}
          disabled={markersRef.current.length == 0 || titleText.length == 0}>
          上传
        </button>
        <textarea placeholder="地点信息" className={s.pointInfo}
          value={markerPicked == null ? "" : markerPicked.getTitle()}
          onChange={(e) => {
            markerPicked?.setTitle(e.target.value);
            forceRender((x) => x + 1);
          }} />
        <button ref={deleteBtnRef} onClick={deletePoint}
          disabled={markerPicked == null}>
          删除
        </button>

        <input type="text" placeholder="搜索" value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { searchBtnRef.current?.click(); } }} />
        <button ref={searchBtnRef} onClick={search}> 搜索 </button>
      </div>
      <div ref={mapRef} className={s.mapContainer} />
    </div>
  );
};

export default MapPinning;
