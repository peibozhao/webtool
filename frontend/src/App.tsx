import { HashRouter, useRoutes } from "react-router-dom";
import s from "./App.module.css";
import SideBar from "./components/SideBar.tsx";
import Home from "./components/Home.tsx";
import Copy from "./components/Copy.tsx";
import Colors from "./components/Colors.tsx";
import Timestamp from "./components/Timestamp.tsx";
import SuperResolution from "./components/SuperResolution.tsx";
import QrCode from "./components/QrCode.tsx";
import AudioRecorder from "./components/AudioRecorder.tsx";
import MapPinning from "./components/MapPinning.tsx"

const router = [
  {
    path: "/",
    get element() {
      return (<Home items={router.filter(({ visible }) => visible).map(({ text, description }) => ({ title: text, description: description }
      ))} />)
    },
    text: "主页",
    description: "各标签页的功能说明",
    visible: true,
  }, {
    path: "/timestamp",
    element: <Timestamp />,
    text: "时间戳",
    description: "unix时间戳跟日期互相转换",
    visible: true,
  }, {
    path: "/colors",
    element: <Colors />,
    text: "颜色表",
    description: "点选颜色, 显示对应的RGB数值",
    visible: true,
  }, {
    path: "/copy",
    element: <Copy />,
    text: "拷贝",
    description: "把长文本保存成便于记忆的短文本, 并可以通过短文本提取长文本(超时时间1天)",
    visible: true,
  }, {
    path: "/super_resolution",
    element: <SuperResolution />,
    text: "超分辨率",
    description: "图像超分辨率(4x)转换",
    visible: false,
  }, {
    path: "/qr_code",
    element: <QrCode />,
    text: "二维码",
    description: "二维码图像的生成和解析",
    visible: true,
  }, {
    path: "/audio_recorder",
    element: <AudioRecorder />,
    text: "录音",
    description: "录音",
    visible: true,
  }, {
    path: "/map_pinning",
    element: <MapPinning />,
    text: "地图打点",
    description: "在地图上进行多次打点记录",
    visible: true,
  }
]

function SideBarInnerApp() {
  return (
    <div className={s.innerApp}>
      {useRoutes(router)}
    </div>
  )
}

function App() {
  return (
    <div className={s.root}>
      <HashRouter>
        <SideBar items={router.filter(({ visible }) => visible).map(({ path, text }) => ({
          to: path, text: text
        }))}>
        </SideBar>

        <SideBarInnerApp>
        </SideBarInnerApp>
      </HashRouter>
    </div>
  )
}

export default App;
