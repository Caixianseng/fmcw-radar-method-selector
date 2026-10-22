"""Self-contained HTML: traffic, time traces, range spectrum and shared-scale STFT."""
from pathlib import Path
from html import escape
from io import BytesIO
import base64,json
import numpy as np
from scipy.signal import stft


def _png(fig):
    output=BytesIO();fig.savefig(output,format='png',dpi=140,bbox_inches='tight',facecolor='white')
    return 'data:image/png;base64,'+base64.b64encode(output.getvalue()).decode()


def render(path, signals, public, rows, outputs, choices, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    records=[];fs=public['sample_rate_hz'];t=np.arange(signals['observed'].shape[-1])/fs*1e6
    for i,row in enumerate(rows):
        clean,x=signals['clean'][i],signals['observed'][i];choice=int(choices[i])
        six=np.stack([clean,x,*outputs[i],outputs[i,choice]])
        f,tt,z=stft(six,fs=fs,window='hann',nperseg=256,noverlap=224,nfft=256,
                    return_onesided=False,boundary='zeros',padded=True)
        db=10*np.log10(np.maximum(abs(np.fft.fftshift(z,axes=-2))**2,1e-32));db-=db[0].max()
        titles=['目标参考','原始接收','MP','C-AR','JSS','MLP → '+('MP','C-AR','JSS')[choice]]
        fig,axes=plt.subplots(2,3,figsize=(13,7));fig.subplots_adjust(wspace=.28,hspace=.4,right=.9)
        for k,ax in enumerate(axes.flat):
            im=ax.pcolormesh(tt*1e6,np.fft.fftshift(f)/1e6,db[k],shading='nearest',cmap='viridis',vmin=-60,vmax=20)
            ax.set(xlim=(0,len(x)/fs*1e6),ylim=(-10,10),xlabel='快时间 / μs',ylabel='频率 / MHz',title=f'({chr(97+k)}) '+titles[k])
        fig.colorbar(im,cax=fig.add_axes([.93,.15,.015,.7]),label='相对目标参考 STFT 峰值 / dB')
        tf=_png(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,3.2))
        for y,label,color in ((x,'原始接收','#2463bf'),(outputs[i,choice],'处理后','#de6e15'),(clean,'目标参考','#17805c')):
            axes[0].plot(t,abs(y)*1e6,label=label,color=color,lw=.8)
            freq=np.fft.fftshift(np.fft.fftfreq(len(y),1/fs))
            range_m=freq*299792458/(2*public['chirp_slope_hz_per_s'])
            spectrum=abs(np.fft.fftshift(np.fft.fft(y*np.hanning(len(y)))))**2/max(np.hanning(len(y)).sum()**2,1)
            axes[1].plot(range_m,10*np.log10(np.maximum(spectrum,1e-32))+30,label=label,color=color,lw=.8)
        axes[0].set(xlabel='快时间 / μs',ylabel='幅度 / μ√W');axes[1].set(xlabel='表观距离 / m',ylabel='距离谱功率 / dBm',xlim=(0,400))
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
        wave=_png(fig);plt.close(fig)
        records.append(dict(info=row,tf=tf,wave=wave,method=titles[-1],scores=scores[i]))
    payload=json.dumps(records,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>FMCW · MLP 恢复对比</title>
<style>body{font:16px/1.6 system-ui;margin:24px auto;max-width:1300px;padding:18px;background:#f3f6fa;color:#172b3f}section{background:white;padding:20px;margin:16px 0;border-radius:10px}img{width:100%}select{padding:8px}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}canvas{width:100%;border:1px solid #ddd}</style>
<h1>四车道 · MLP 方法选择与时频恢复</h1><p>六台干扰雷达；每会话1/3概率配置至多一个近斜率源；1000 m；77 GHz / 300 MHz / 50 MHz / 3750点。</p>
<section><label>逐条 chirp <select id="sample"></select></label><p id="info"></p><canvas id="road" width="1200" height="230"></canvas><details><summary>车辆与独立时钟参数</summary><pre id="scene"></pre></details></section>
<section><h2>时频恢复六联图</h2><p>所有面板共用色标和坐标；目标参考仅用于离线比较，MLP实际输出与选中候选一致。</p><img id="tf"></section>
<section><h2>恢复波形与距离谱</h2><img id="wave"><table id="scores"></table></section>
<script>const rows=__PAYLOAD__,sel=document.getElementById('sample');rows.forEach((r,i)=>sel.add(new Option('seed '+r.info.seed+' · '+r.info.window_s+' s · chirp '+r.info.chirp,i)));
function draw(){const r=rows[sel.value||0];document.getElementById('tf').src=r.tf;document.getElementById('wave').src=r.wave;document.getElementById('info').textContent=r.method+'；目标路径 '+r.info.target_paths+'；'+(r.info.configured_extreme_event?'配置近斜率源':'普通源配置');document.getElementById('scene').textContent=JSON.stringify(r.info.scene,null,2);document.getElementById('scores').innerHTML='<tr><th>输出</th><th>SINR / dB</th><th>线性 NMSE</th></tr>'+r.scores.map(s=>'<tr><td>'+s.method+'</td><td>'+s.sinr_db.toFixed(2)+'</td><td>'+s.nmse.toFixed(5)+'</td></tr>').join('');const c=document.getElementById('road'),ctx=c.getContext('2d');
ctx.fillStyle='#edf2f6';ctx.fillRect(0,0,c.width,c.height);
ctx.strokeStyle='#789';for(let l=0;l<5;l++){ctx.beginPath();ctx.moveTo(0,15+l*49);ctx.lineTo(c.width,15+l*49);ctx.stroke();}
const scene=r.info.scene||{},ego=scene.ego||{},cars=[ego,...(scene.vehicles||[])].filter(v=>Number.isFinite(v.y_m));
const rel=cars.map(v=>v.y_m-ego.y_m),lo=Math.min(-25,...rel)-15,hi=Math.max(25,...rel)+15;
ctx.fillStyle='#172b3f';ctx.font='12px sans-serif';ctx.fillText('车辆位置按道路纵向坐标映射；相对距离以自车为零点',20,12);
cars.forEach(v=>{const dx=v.y_m-ego.y_m,x=45+(dx-lo)/(hi-lo)*(c.width-100),lane=Math.max(0,Math.min(3,v.lane)),y=28+lane*49;
ctx.fillStyle=v.is_ego?'#195f9c':(v.radar?'#d56a18':'#888');ctx.fillRect(x-18,y,36,19);ctx.fillStyle='#172b3f';ctx.fillText((v.is_ego?'自车':'V'+v.vehicle_id)+' '+dx.toFixed(1)+'m',x-30,y+34);});
}sel.onchange=draw;draw();</script></html>'''
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(page.replace('__PAYLOAD__',payload),encoding='utf-8')
