// Each dot corresponds to one archived PR 6 query. The table is its keyboard-accessible equivalent.
export function scorePlot(canvas, choose) {
  let points = [], selected, coordinates = [];
  function paint() {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    ctx.scale(scale,scale);
    const left = 46, right = width-14, top = 16, bottom = height-36;
    const x = score => left + score*(right-left);
    const y = gap => bottom - gap*(bottom-top);
    ctx.font = "12px system-ui";
    ctx.textBaseline = "middle";
    for (const value of [0,.25,.5,.75,1]) {
      ctx.strokeStyle="#d6dbcd";
      ctx.beginPath();ctx.moveTo(left,y(value));ctx.lineTo(right,y(value));ctx.stroke();
      ctx.fillStyle="#435244";ctx.textAlign="right";ctx.fillText(value.toFixed(2),left-5,y(value));
      ctx.textAlign="center";ctx.fillText(value.toFixed(2),x(value),bottom+14);
    }
    ctx.strokeStyle="#253320";ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(x(.749896646),top);ctx.lineTo(x(.749896646),bottom);
    ctx.moveTo(left,y(.117484212));ctx.lineTo(right,y(.117484212));ctx.stroke();ctx.setLineDash([]);
    coordinates=points.map(q=>({q,x:x(q.score),y:y(q.gap)}));
    for (const p of coordinates) {
      ctx.fillStyle=p.q.known?"#36896988":"#b7694688";
      ctx.beginPath();ctx.arc(p.x,p.y,2.4,0,Math.PI*2);ctx.fill();
    }
    if (selected) {
      ctx.strokeStyle="#15221b";ctx.lineWidth=2;ctx.beginPath();
      ctx.arc(x(selected.score),y(selected.gap),6,0,Math.PI*2);ctx.stroke();
    }
  }
  canvas.addEventListener("click", event=>{
    const rect=canvas.getBoundingClientRect(), x=event.clientX-rect.left, y=event.clientY-rect.top;
    let closest, best=100;
    for(const p of coordinates) {
      const distance=(p.x-x)**2+(p.y-y)**2;
      if(distance<best){best=distance;closest=p.q;}
    }
    if(closest) choose(closest);
  });
  new ResizeObserver(paint).observe(canvas);
  return (rows, current)=>{points=rows;selected=current;paint();};
}

