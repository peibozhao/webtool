import { useState } from 'react';
import { useTitle } from '../hooks/useTitle';
import { notification } from "antd"
import { backendServer, processFetchResponse } from '../common/utils';
import s from './Copy.module.css';

function Copy() {
  useTitle('拷贝');

  const [submitText, setSubmitText] = useState('');
  const [submitButtonText, setSubmitButtonText] = useState('提交');
  const [retrieveCode, setRetrieveCode] = useState('');
  const [retrieveButtonText, setRetrieveButtonText] = useState('提取');

  const [notifyApi, notifyContext] = notification.useNotification();

  const backend_server = backendServer();

  const submitHandler = async () => {
    setSubmitButtonText('提交中...');
    try {
      const response = await fetch(`${backend_server}/api/copy/submit`, {
        method: 'POST',
        body: `text=${submitText}`,
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });
      if (await processFetchResponse(response, notifyApi, '提交')) {
        const data = await response.json();
        setRetrieveCode(data.code);
      }
      setSubmitButtonText('提交');
    } catch (error) {
      setSubmitButtonText('提交');
      notifyApi.error({ message: '提交失败', duration: 3 });
      console.error("异常: ", error);
    }
    setSubmitText('');
  };

  const retrieveHandler = async () => {
    setRetrieveButtonText('提取中...');
    try {
      const response = await fetch(`${backend_server}/api/copy/retrieve?code=${retrieveCode}`);
      if (await processFetchResponse(response, notifyApi, '提取')) {
        const data = await response.json();
        setSubmitText(data.text);
      }
      setRetrieveButtonText('提取');
    } catch (error) {
      setRetrieveButtonText('提取');
      notifyApi.error({ message: '提取失败', duration: 3 });
      console.error("异常: ", error);
    }
    setRetrieveCode('');
  }

  return (
    <div>
      {notifyContext}
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

export default Copy;
