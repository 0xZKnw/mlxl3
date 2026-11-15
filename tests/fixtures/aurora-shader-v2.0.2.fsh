vec2 lightUV(vec2 p, float t) {
  vec2 drift = vec2(0.014*(sin(4.0*p.y+0.44*t)-sin(4.0*p.y)),
                    0.028*(sin(5.0*p.x-0.49*t)-sin(5.0*p.x)));
  vec2 at = clamp(p+drift*smoothstep(vec2(0.0),vec2(0.12),min(p,1.0-p)),0.0,1.0);
  return vec2(at.x,1.0-at.y);
}
void main() {
  vec2 p = vec2(v_tex_coord.x,1.0-v_tex_coord.y);
  float t = phase;
  vec2 cell = floor(p*cells);
  vec2 center = (cell+0.5)/cells;
  vec2 local = fract(p*cells);
  float arc = p.y + 0.065*sin(5.0*p.x + 0.24*t) + 0.035*sin(10.0*p.x - 0.31*t);
  float ribbon = 0.5 + 0.5*cos(10.5*arc - 0.82*t + 1.3*p.x);
  float glint = 0.5 + 0.5*sin(18.0*arc + 3.3*p.x - 0.58*t);
  float veil = 0.5 + 0.5*cos(4.7*p.x - 2.6*p.y + 0.37*t);
  float light = 0.77 + 0.26*ribbon + 0.12*glint*glint + 0.05*veil;
  vec3 emission = vec3(light + 0.08*ribbon*glint, light, light + 0.085*(1.0-ribbon)*veil);
  vec4 lamp = texture2D(u_texture,lightUV(p,t));
  float intensity = texture2D(u_texture,lightUV(center,t)).a;
  float wave = 0.5+0.5*cos(10.5*(center.y+0.065*sin(5.0*center.x+0.24*t))-0.82*t+1.3*center.x);
  float level = 6.0*clamp(intensity*(0.65+0.60*wave),0.0,1.0);
  float index = floor(level);
  vec2 tile = vec2(local.x/7.0,1.0-local.y);
  float first = texture2D(alphabet,tile+vec2(index/7.0,0.0)).a;
  float next = texture2D(alphabet,tile+vec2(min(6.0,index+1.0)/7.0,0.0)).a;
  float mark = mix(first,next,smoothstep(0.32,0.68,fract(level)));
  float alpha = lamp.a;
  vec3 color = min(lamp.rgb*emission,vec3(alpha));
  gl_FragColor = vec4(mix(color,vec3(alpha),0.45*mark*smoothstep(0.25,1.65,level)),alpha);
}