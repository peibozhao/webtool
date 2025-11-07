import { useState, useRef } from "react";
import { notification } from "antd"
import { backendServer, processFetchResponse } from '../common/utils';

function QrCode() {
  const [text, setText] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [notifyApi, notifyContext] = notification.useNotification();

  const backend_server = backendServer();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    if (selectedFile) {
      setImageUrl(URL.createObjectURL(selectedFile));
      setImageFile(selectedFile);
    }
  };

  const handleSubmitText = async () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    try {
      const response = await fetch(`${backend_server}/api/qr_code/generate`, {
        method: "POST",
        body: `text=${text}`,
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });
      if (await processFetchResponse(response, notifyApi, '提交')) {
        const data = await response.blob();
        setImageUrl(URL.createObjectURL(data));
      }
    } catch (error) {
      setImageUrl('');
      notifyApi.error({ message: '提交失败', duration: 3 });
      console.error("异常: ", error);
    }
  };

  const handleUploadImage = async () => {
    if (!imageFile) return;

    const formData = new FormData();
    formData.append("image", imageFile);
    setText('');
    try {
      const response = await fetch(`${backend_server}/api/qr_code/parse`, {
        method: "POST",
        body: formData,
      });
      if (await processFetchResponse(response, notifyApi, '监测')) {
        const data = await response.json();
        setText(data.text)
      }
    } catch (error) {
      notifyApi.error({ message: '监测失败', duration: 3 });
      console.error("异常: ", error);
    }
  };

  return (
    <div>
      {notifyContext}
      <div>
        <input type="text" value={text} onChange={(e) => setText(e.target.value)} />
        <button onClick={handleSubmitText}>生成</button>
      </div>
      <div>
        <input type="file" accept="image/*" onChange={handleFileChange} ref={fileInputRef} />
        <button onClick={handleUploadImage}>解析</button>
      </div>
      {imageUrl && <img src={imageUrl} />}
    </div>
  );
}

export default QrCode;

