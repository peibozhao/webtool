
import { useState } from 'react'
import { useTitle } from '../hooks/useTitle'
import s from './Copy.module.css'

function Copy() {
  useTitle('拷贝')

  const [submitText, setSubmitText] = useState('')
  const [submitButtonText, setSubmitButtonText] = useState('提交')
  const [retrieveCode, setRetrieveCode] = useState('')
  const [retrieveButtonText, setRetrieveButtonText] = useState('提取')

  const backend_server = import.meta.env.VITE_BACKEND_SERVER

  const submitHandler = () => {
    setSubmitButtonText('提交中...')
    fetch(`${backend_server}/api/copy/submit`, {
      method: 'POST',
      body: `text=${submitText}`,
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    }).then((res) => {
      return res.json()
    }).then((data) => {
      setRetrieveCode(data.code)
    }).catch((err) => {
      alert(err.message)
    }).finally(() => {
      setSubmitButtonText('提交')
    })
    setSubmitText('')
  };

  const retrieveHandler = () => {
    setRetrieveButtonText('提取中...')
    fetch(`${backend_server}/api/copy/retrieve?code=${retrieveCode}`).then((res) => {
      return res.json()
    }).then((data) => {
      setSubmitText(data.text)
    }).catch((err) => {
      alert(err.message)
    }).finally(() => {
      setRetrieveButtonText('提取')
    });
    setRetrieveCode('')
  }

  return (
    <div>
      <div className={s.row}>
        <input type='text' value={submitText} onChange={(e) => setSubmitText(e.target.value)} />
        <button onClick={submitHandler}>
          {submitButtonText}
        </button>
      </div>
      <div className={s.row}>
        <input type='text' value={retrieveCode} onChange={(e) => setRetrieveCode(e.target.value)} />
        <button onClick={retrieveHandler}>
          {retrieveButtonText}
        </button>
      </div>
    </div>
  )
}

export default Copy

