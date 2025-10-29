
import s from "./Home.module.css"

interface HomeProps {
  items: { title: string, description: string }[];
}

function Home({ items }: HomeProps) {
  return <div className={s.root}>
    <table className={s.descriptionTable}>
      {
        items.map(({ title, description }) => (
          <tr className={s.row}>
            <td className={s.titleCol}> {title} </td>
            <td className={s.descriptionCol}> {description} </td>
          </tr>
        ))
      }
    </table>
  </div>
}

export default Home;

