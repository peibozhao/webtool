import s from "./Home.module.css"

interface HomeProps {
  items: { title: string, description: string }[];
}

function Home({ items }: HomeProps) {
  return <div className={s.root}>
    <table className={s.descriptionTable}>
      <tbody>
        {
          items.map(({ title, description }) => (
            <tr key={title} className={s.row}>
              <td className={s.titleCol}> {title} </td>
              <td className={s.descriptionCol}> {description} </td>
            </tr>
          ))
        }
      </tbody>
    </table>
  </div>
}

export default Home;

