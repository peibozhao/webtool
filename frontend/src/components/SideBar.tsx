
import { NavLink } from 'react-router-dom'
import s from './SideBar.module.css'
import clsx from 'clsx'

interface SideBarProps {
  items: { to: string, text: string }[];
}

function SideBar({ items }: SideBarProps) {
  return (
    <div className={s.root}>
      <div className={s.rows}>
        {
          items.map(({ to, text }) => (
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

