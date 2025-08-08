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
    element: <Home />
  }, {
    path: '/copy',
    element: <Copy />
  }, {
    path: '/colors',
    element: <Colors />
  }, {
    path: '/timestamp',
    element: <Timestamp />
  }, {
    path: '/test',
    element: <Test />
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
        <SideBar>
        </SideBar>

        <SideBarInnerApp>
        </SideBarInnerApp>
      </HashRouter>
    </div>
  )
}

export default App
