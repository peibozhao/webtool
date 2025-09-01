import { HashRouter, useRoutes } from 'react-router-dom'
import s from './App.module.css'
import SideBar from './components/SideBar.tsx'
import Home from './components/Home.tsx'
import Copy from './components/Copy.tsx'
import Colors from './components/Colors.tsx'
import Timestamp from './components/Timestamp.tsx'
import Test from './components/Test.tsx'

const router = [
  {
    path: '/',
    element: <Home />,
    text: '主页',
    description: '',
    visible: true,
  }, {
    path: '/copy',
    element: <Copy />,
    text: '拷贝',
    description: '',
    visible: true,
  }, {
    path: '/colors',
    element: <Colors />,
    text: '颜色表',
    description: '',
    visible: true,
  }, {
    path: '/timestamp',
    element: <Timestamp />,
    text: '时间戳',
    description: '',
    visible: true,
  }, {
    path: '/test',
    element: <Test />,
    text: '测试页面',
    description: '',
    visible: false,
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

export default App
