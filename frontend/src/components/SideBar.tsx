
import { NavLink } from 'react-router-dom'
import s from './SideBar.module.css'
import clsx from 'clsx'

const pages = [
  { to: '/colors', text: '颜色表' },
  { to: '/timestamp', text: '时间戳' },
  { to: '/copy', text: '拷贝' },
]

function SideBar() {
  return (
    <div className={s.root}>
      <div className={s.rows}>
        {
          pages.map(({ to, text }) => (
            <NavLink to={to} key={to}
              className={({ isActive }) => clsx(s.row, isActive ? s.rowActive : null)}
            >
              <div className={s.label}> {text} </div>
            </NavLink>
          ))
        }
      </div>
    </div>
  )
}

export default SideBar

