import React, { useState, useRef, useEffect } from "react";
import { useTitle } from '../hooks/useTitle';
import s from './Colors.module.css';

function ColorValueLimit(value: number) {
  return Math.min(255, Math.max(0, Math.floor(value)));
}

function Colors() {
  useTitle('颜色表');

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [y, setY] = useState(125);
  const [u, setU] = useState('');
  const [v, setV] = useState('');
  const [colorRGB, setColorRGB] = useState({ r: '', g: '', b: '' });

  const width = 256;
  const height = 256;

  const drawColors = (ctx: CanvasRenderingContext2D) => {
    const img = ctx.createImageData(width, height);
    let index = 0;
    for (let h = 0; h < height; h += 1) {
      for (let w = 0; w < width; w += 1) {
        const r = ColorValueLimit(y + 1.402 * (w - 128));
        const g = ColorValueLimit(y - 0.344136 * (h - 128) - 0.714136 * (w - 128));
        const b = ColorValueLimit(y + 1.772 * (h - 128));

        img.data[index++] = r;
        img.data[index++] = g;
        img.data[index++] = b
        img.data[index++] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawColors(ctx);
  });

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * width);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * height);

    setU(String(y));
    setV(String(x));

    const pixel = ctx.getImageData(x, y, 1, 1).data;
    setColorRGB({ r: String(pixel[0]), g: String(pixel[1]), b: String(pixel[2]) })
  };

  const brightnessChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newY = Number(e.target.value);
    setY(newY);

    if (colorRGB.r.length == 0) {
      return;
    }

    const r = ColorValueLimit(newY + 1.402 * (Number(v) - 128));
    const g = ColorValueLimit(newY - 0.344136 * (Number(u) - 128) - 0.714136 * (Number(v) - 128));
    const b = ColorValueLimit(newY + 1.772 * (Number(u) - 128));

    setColorRGB({ r: String(r), g: String(g), b: String(b) });
  };

  return (
    <div className={s.root}>
      <div className={s.header}>
        <div className={s.headerValue}>
          <div>
            <span>R:</span>
            <input className={s.value} type='text' value={colorRGB.r} readOnly />
            <span>G:</span>
            <input className={s.value} type='text' value={colorRGB.g} readOnly />
            <span>B:</span>
            <input className={s.value} type='text' value={colorRGB.b} readOnly />
          </div>
          <div>
            <span>Y:</span>
            <input type='range' className={s.yValue} min='0' max='255' value={y} onChange={(e) => brightnessChange(e)} />
            <span>U:</span>
            <input className={s.value} type='text' value={u} readOnly />
            <span>V:</span>
            <input className={s.value} type='text' value={v} readOnly />
          </div>
        </div>
        <div className={s.colorPicked} style={{ backgroundColor: `rgb(${colorRGB.r}, ${colorRGB.g}, ${colorRGB.b})` }}>
        </div>
      </div>
      <canvas
        className={s.colors}
        ref={canvasRef}
        onClick={handleClick}
        width={width}
        height={height}
      />
    </div>
  );
};

export default Colors;

