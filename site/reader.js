"use strict";
(()=>{
  const toggle=document.querySelector('.menu-toggle');
  const close=()=>{document.body.classList.remove('menu-open');toggle?.setAttribute('aria-expanded','false');};
  toggle?.addEventListener('click',()=>{const open=document.body.classList.toggle('menu-open');toggle.setAttribute('aria-expanded',String(open));});
  document.addEventListener('keydown',event=>{if(event.key==='Escape')close();});
  const links=[...document.querySelectorAll('.site-sidebar a[href]')];
  function current(){
    const here=new URL(location.href);
    for(const link of links){
      const target=new URL(link.href);
      const active=target.pathname.replace(/index\.html$/,'')===here.pathname.replace(/index\.html$/,'')&&target.hash===here.hash;
      if(active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');
    }
  }
  addEventListener('hashchange',current);current();
  document.querySelector('.site-sidebar')?.addEventListener('click',event=>{if(innerWidth<=800&&event.target.closest('a'))close();});
})();
