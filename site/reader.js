"use strict";
(()=>{
  const toggle=document.querySelector('.menu-toggle');
  toggle?.addEventListener('click',()=>{const open=document.body.classList.toggle('menu-open');toggle.setAttribute('aria-expanded',String(open));});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.body.classList.remove('menu-open');toggle?.setAttribute('aria-expanded','false');}});
  const links=[...document.querySelectorAll('.site-sidebar a[href]')];
  function current(){
    const here=new URL(location.href),page=here.searchParams.get('page');
    const course=here.searchParams.get('course')||document.body.dataset.course;
    for(const a of links){
      const u=new URL(a.href),samePath=u.pathname.replace(/index\.html$/,'')===here.pathname.replace(/index\.html$/,'');
      const active=samePath&&(page?u.searchParams.get('page')===page&&(!u.searchParams.has('course')||u.searchParams.get('course')===course):u.hash===here.hash);
      if(active)a.setAttribute('aria-current',page?'page':'location');else a.removeAttribute('aria-current');
    }
    if(course){const group=document.querySelector('[data-course-nav="'+course+'"]');if(group)group.open=true;}
  }
  window.StudyNavigation={refresh:current};
  addEventListener('hashchange',current);current();
  document.querySelector('.site-sidebar')?.addEventListener('click',e=>{if(innerWidth<=800&&e.target.closest('a,button')){document.body.classList.remove('menu-open');toggle?.setAttribute('aria-expanded','false');}});
})();
