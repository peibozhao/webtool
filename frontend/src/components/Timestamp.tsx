
import { useState } from 'react';
import { DatePicker, TimePicker, Switch } from 'antd';
import dayjs from 'dayjs';
import { useTitle } from '../hooks/useTitle';
import s from './Timestamp.module.css';

type TimestampUnit = 's' | 'ms' | 'us' | 'ns';

const timeUnitsDemnominator: { [key in TimestampUnit]: number } = {
  s: 1,
  ms: 1000,
  us: 1000000,
  ns: 1000000000,
};

function Timestamp() {
  useTitle('时间戳');

  const [timestamp, setTimestamp] = useState(String(dayjs().unix()));
  const [timestampUnit, setTimestampUnit] = useState<TimestampUnit>('s');
  const [autoUnit, setAutoUnit] = useState(true);

  const timeChange = (date: dayjs.Dayjs, _: any) => {
    setTimestamp(String(date.unix()));
  };

  const timestampChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    if (!/^\d*\.?\d*$/.test(newValue)) {
      return;
    }
    setTimestamp(newValue);
    if (!autoUnit) {
      return;
    }
    // Auto choose unit
    if (newValue.length <= 10) {
      setTimestampUnit('s');
    } else if (newValue.length <= 16) {
      setTimestampUnit('ms');
    } else if (newValue.length <= 19) {
      setTimestampUnit('us');
    } else {
      setTimestampUnit('ns');
    }
  };

  const autoUnitChange = (e: boolean) => {
    setAutoUnit(e);
    if (!e) {
      return;
    }
    // Auto choose unit
    if (timestamp.length <= 10) {
      setTimestampUnit('s');
    } else if (timestamp.length <= 13) {
      setTimestampUnit('ms');
    } else if (timestamp.length <= 16) {
      setTimestampUnit('us');
    } else {
      setTimestampUnit('ns');
    }
  }

  return (
    <div>
      <div className={s.row}>
        <span>时间:</span>
        <DatePicker className={s.picker} value={dayjs.unix(Number(timestamp) / timeUnitsDemnominator[timestampUnit])} onChange={timeChange} />
        <TimePicker className={s.picker} value={dayjs.unix(Number(timestamp) / timeUnitsDemnominator[timestampUnit])} onChange={timeChange} />
      </div>
      <div className={s.row}>
        <span>时间戳:</span>
        <input type='text' value={timestamp} onChange={timestampChange} />
        <select value={timestampUnit} onChange={(e) => setTimestampUnit(e.target.value as TimestampUnit)}>
          {
            Object.keys(timeUnitsDemnominator).map((unit: string) => {
              return <option value={unit} key={unit}>{unit}</option>
            })
          }
        </select>
        <Switch className={s.auto} value={autoUnit} onChange={autoUnitChange} />
      </div>
    </div>
  );
};

export default Timestamp;

